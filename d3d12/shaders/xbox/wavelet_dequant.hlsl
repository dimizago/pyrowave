// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// Dequantizer for the moonlight-xbox client on Xbox Series. Not a general-purpose
// kernel: it is tuned for the GPU partition that client runs in (a UWP declaring
// hevcPlayback, which it needs for HDR10 output, gets the console's "4K media app"
// share: ~8% of the ALU throughput, ~116 GB/s, LDS cut like ALU, ~5x latency; see
// xbox_kernel_model in pyrowave_d3d12_device.cpp), and for the rates it streams at.
// There it takes a 4K 4:4:4 frame at ~460 KB from 2.1 ms (the translated
// shaders/wavelet_dequant.comp) to ~0.05 ms. On desktop GPUs it is no faster, and the
// device only uses it where waves are always 64 lanes, i.e. on the Xbox.
//
// Decodes the same coefficients as shaders/wavelet_dequant.comp, bit for bit before
// the output changes described below.
//
// One wave per 32x32 block (the GLSL version uses 128 threads, one 4x2 sub-block each).
// Lane l decodes sub-blocks 2l and 2l+1 in the GLSL thread order, which are vertically
// adjacent, so a lane owns a 4x4 area and the sign bits of its two sub-blocks follow
// each other in the bitstream: one wave prefix sum places every lane's signs.
//
// The decoded tile is staged in LDS and written out one 8x8 square per store
// instruction: on the Xbox's media-app GPU partition, typed stores whose 64 lanes are
// spread over the tile (as each thread writing its own 4x2 area is) run about 8x
// slower than compact ones, and that, not the decoding, dominated the GLSL version.
//
// Per sub-block, the n plane bytes (MSB plane first) are fetched with one 12-byte load
// and turned into eight magnitudes by an 8x8 bit-matrix transpose, instead of a
// dependent byte load plus sixteen bit operations per plane. Sub-blocks with more than
// seven planes (rare: the low-frequency bands at high rates) take a plain loop.
//
// Unlike the GLSL shader, HH coefficients are stored divided by K^2 and the coarsest LL
// band multiplied by K^2: the Xbox iDWT (xbox/idwt.hlsl) expects the CDF 9/7 input
// scaling to be folded in here. And with skip_empty_blocks (the two finest levels) the
// empty 32x32 blocks (no coefficients at all; at streaming rates nearly all of them)
// are not written: the iDWT looks them up in the block offset table instead. Values
// are rounded to FP16 to nearest even before the store, since the hardware conversion
// into the FP16 pyramid truncates; that truncation was the largest error in the
// translated decoder's output.
//
// Compiled for Shader Model 6.4 (Xbox, where every wave has 64 lanes), and with
// [WaveSize(64)] for Shader Model 6.6 to test on a PC GPU; see transpile.py.

#ifndef PW_WAVE_SIZE_ATTRIBUTE
#define PW_WAVE_SIZE_ATTRIBUTE
#endif

cbuffer Registers : register(b0)
{
	int2 resolution;
	int output_layer;
	int block_offset_32x32;
	int block_stride_32x32;
	// Nonzero: leave empty blocks unwritten (the iDWT of this level checks for them).
	uint skip_empty_blocks;
};

ByteAddressBuffer Offsets : register(t1);
ByteAddressBuffer Payload : register(t5);
RWTexture2DArray<float> Pyramid : register(u0);

// Per 8x8 block: first plane byte, control word | q byte << 16.
groupshared uint2 g_block[16];
// PDEP of a 4-bit sign run into a 4-bit nonzero mask, indexed by mask | signs << 4.
groupshared uint g_pdep[256];
// The decoded 32x32 tile, padded against bank conflicts.
groupshared float g_tile[32][33];

static const uint PDEP4_PACKED[64] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x01020100, 0x01020104, 0x01020108, 0x01020104,
	0x02000000, 0x02040400, 0x02080800, 0x02040408, 0x03020100, 0x03060504, 0x030a0908, 0x0306050c,
	0x00000000, 0x04000000, 0x08000000, 0x04080800, 0x01020100, 0x05020104, 0x09020108, 0x050a0904,
	0x02000000, 0x06040400, 0x0a080800, 0x060c0c08, 0x03020100, 0x07060504, 0x0b0a0908, 0x070e0d0c,
	0x00000000, 0x00000000, 0x00000000, 0x08000000, 0x01020100, 0x01020104, 0x01020108, 0x09020104,
	0x02000000, 0x02040400, 0x02080800, 0x0a040408, 0x03020100, 0x03060504, 0x030a0908, 0x0b06050c,
	0x00000000, 0x04000000, 0x08000000, 0x0c080800, 0x01020100, 0x05020104, 0x09020108, 0x0d0a0904,
	0x02000000, 0x06040400, 0x0a080800, 0x0e0c0c08, 0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
};

// Same formulations as the GLSL shader, so the float results are identical.
float decode_quant(uint quant_code)
{
	int e = 4 - int(quant_code >> 3u);
	int m = int(quant_code) & 7;
	return 1.1920928955078125e-07f * float((8 + m) * (1 << (20 + e)));
}

float decode_quant_scale(uint code)
{
	return (float(code) / 8.0f) + 0.25f;
}

uint load_u8(uint byte_address)
{
	return (Payload.Load(byte_address & ~3u) >> ((byte_address & 3u) * 8u)) & 0xffu;
}

uint load_u16(uint byte_address)
{
	return (Payload.Load(byte_address & ~3u) >> ((byte_address & 2u) * 8u)) & 0xffffu;
}

// Bits [shift, shift + 32) of the 64-bit hi:lo, shift in [0, 32).
uint funnel(uint lo, uint hi, uint shift)
{
	return (lo >> shift) | ((hi << 1u) << (31u - shift));
}

struct SubBlock
{
	// 2 * magnitude + 1 for nonzero coefficients, 0 otherwise; coefficient c is at
	// x = c / 2, y = c % 2 of the 4x2 sub-block.
	float mag2[8];
	uint nz_lo; // nonzero mask of coefficients 0-3
	uint nz_hi; // nonzero mask of coefficients 4-7
	uint count;
};

// Compresses bytes' top bits (0x80 per nonzero byte) into a 4-bit mask.
uint compress_nz(uint m)
{
	uint x = m >> 7u;
	return (x | (x >> 7u) | (x >> 14u) | (x >> 21u)) & 15u;
}

SubBlock decode(uint pos, uint n)
{
	SubBlock r;
	[unroll] for (int c = 0; c < 8; c++)
		r.mag2[c] = 0.0f;
	r.nz_lo = 0;
	r.nz_hi = 0;
	r.count = 0;

	if (n == 0)
		return r;

	if (n <= 7)
	{
		// The 8 bytes ending with the last plane byte: planes P0..P(n-1) land in bytes
		// 8-n..7, preceded by unrelated bytes which are masked off.
		const uint first = pos + n - 8u;
		const uint3 w = Payload.Load3(first & ~3u);
		const uint shift = (first & 3u) * 8u;
		uint lo = funnel(w.x, w.y, shift);
		uint hi = funnel(w.y, w.z, shift);
		const uint keep = 64u - 8u * n; // low bits to clear, 8..56
		lo = keep >= 32u ? 0u : lo & (0xffffffffu << keep);
		hi = keep >= 32u ? hi & (0xffffffffu << (keep - 32u)) : hi;

		// 8x8 bit transpose: byte k bit c -> byte c bit k (Hacker's Delight transpose8).
		uint t = (lo ^ (lo >> 7u)) & 0x00aa00aau;
		lo ^= t ^ (t << 7u);
		t = (hi ^ (hi >> 7u)) & 0x00aa00aau;
		hi ^= t ^ (t << 7u);
		t = (lo ^ (lo >> 14u)) & 0x0000ccccu;
		lo ^= t ^ (t << 14u);
		t = (hi ^ (hi >> 14u)) & 0x0000ccccu;
		hi ^= t ^ (t << 14u);
		t = (lo ^ (hi << 4u)) & 0xf0f0f0f0u;
		lo ^= t;
		hi ^= t >> 4u;

		// Byte c now holds coefficient c's bits with plane Pj at bit j + 8 - n, so
		// reversing it yields the magnitude MSB-first. bfrev reverses whole words,
		// which also reverses the byte order: coefficient c ends up in byte 3 - c % 4.
		const uint m_lo = (((lo & 0x7f7f7f7fu) + 0x7f7f7f7fu) | lo) & 0x80808080u;
		const uint m_hi = (((hi & 0x7f7f7f7fu) + 0x7f7f7f7fu) | hi) & 0x80808080u;
		r.nz_lo = compress_nz(m_lo);
		r.nz_hi = compress_nz(m_hi);
		r.count = countbits(m_lo) + countbits(m_hi);

		// 2a + 1 for nonzero a (a < 128 with at most 7 planes), 0 for zero.
		const uint a2_lo = (reversebits(lo) << 1u) | reversebits(m_lo);
		const uint a2_hi = (reversebits(hi) << 1u) | reversebits(m_hi);
		[unroll] for (uint i = 0; i < 4; i++)
		{
			r.mag2[i] = float((a2_lo >> (24u - 8u * i)) & 0xffu);
			r.mag2[i + 4] = float((a2_hi >> (24u - 8u * i)) & 0xffu);
		}
		return r;
	}

	// Many planes: bit-serial.
	uint a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	for (uint k = 0; k < n; k++)
	{
		const uint plane = load_u8(pos + k);
		[unroll] for (uint b = 0; b < 8; b++)
			a[b] = (a[b] << 1u) | ((plane >> b) & 1u);
	}
	[unroll] for (uint i = 0; i < 8; i++)
	{
		const bool nz = a[i] != 0;
		r.mag2[i] = nz ? float(2u * a[i] + 1u) : 0.0f;
		if (i < 4)
			r.nz_lo |= uint(nz) << i;
		else
			r.nz_hi |= uint(nz) << (i - 4);
		r.count += uint(nz);
	}
	return r;
}

// Applies the sign bits starting at bit sign_offset and the scale, and puts the 4x2
// sub-block at tile position coord.
void finish(SubBlock s, uint sign_offset, float half_scale, uint2 coord)
{
	const uint2 words = Payload.Load2((sign_offset >> 5u) * 4u);
	const uint signs = funnel(words.x, words.y, sign_offset & 31u);

	// Spread the sign run over the nonzero coefficients.
	const uint expanded = g_pdep[s.nz_lo | ((signs & 15u) << 4u)] |
	                      (g_pdep[s.nz_hi | (((signs >> countbits(s.nz_lo)) & 15u) << 4u)] << 4u);

	const uint scale_bits = asuint(half_scale);
	[unroll] for (uint c = 0; c < 8; c++)
	{
		// (a + 0.5) * scale, negated per the sign bit, rounded once as in the GLSL.
		g_tile[coord.y + (c & 1u)][coord.x + (c >> 1u)] =
				s.mag2[c] * asfloat(scale_bits | ((expanded << (31u - c)) & 0x80000000u));
	}
}

// FP32 -> FP16 rounding to nearest even (see xbox/idwt.hlsl): the pyramid store's own
// conversion would truncate.
float round_to_fp16(float v)
{
	uint b = asuint(v);
	b += 0xfffu + ((b >> 13u) & 1u);
	return f16tof32(f32tof16(asfloat(b & ~0x1fffu)));
}

// Writes the tile, each store instruction covering one 8x8 square.
void store_tile(uint2 group, uint lane, bool zero)
{
	const int2 base = int2(group) * 32 + int2(lane & 7u, lane >> 3u);
	[unroll] for (uint k = 0; k < 16; k++)
	{
		const uint2 p = uint2((k & 3u) * 8u + (lane & 7u), (k >> 2u) * 8u + (lane >> 3u));
		Pyramid[int3(base + 8 * int2(k & 3u, k >> 2u), output_layer)] = zero ? 0.0f : round_to_fp16(g_tile[p.y][p.x]);
	}
}

[numthreads(64, 1, 1)]
PW_WAVE_SIZE_ATTRIBUTE
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
	// Sub-blocks 2l and 2l+1: 8x8 block l / 4, rows 4 * (l % 2) .. + 3 of column
	// 4 * ((l / 2) % 2) within it.
	const uint blk = lane >> 2u;
	const uint sub = (lane & 3u) * 2u;
	const uint2 origin = uint2(8u * (blk & 3u) + 4u * (sub >> 2u), 8u * (blk >> 2u) + 2u * (sub & 3u));

	const uint block_index = uint(block_offset_32x32) + group.y * uint(block_stride_32x32) + group.x;
	const uint offset = Offsets.Load(block_index * 4u);
	if (offset == 0xffffffffu)
	{
		if (skip_empty_blocks == 0)
			store_tile(group.xy, lane, true);
		return;
	}

	const uint base = offset * 4u;
	const uint2 header = Payload.Load2(base);
	const uint ballot = header.x & 0xffffu;
	const uint q_code = header.y & 0xffu;
	const uint active_blocks = countbits(ballot);

	// Lanes 0-15 read the control data of 8x8 block `lane` and size its plane data.
	uint cost = 0, info = 0;
	if (lane < 16u && ((ballot >> lane) & 1u) != 0)
	{
		const uint k = countbits(ballot & ((1u << lane) - 1u));
		const uint control = load_u16(base + 8u + 2u * k);
		const uint q_byte = load_u8(base + 8u + 2u * active_blocks + k);
		cost = countbits(control & 0x5555u) + 2u * countbits(control & 0xaaaau) + 8u * (q_byte & 15u);
		info = control | (q_byte << 16u);
	}
	const uint planes_start = base + 8u + 3u * active_blocks;
	const uint block_start = planes_start + WavePrefixSum(cost);
	const uint sign_base = 8u * (planes_start + WaveActiveSum(cost));
	if (lane < 16u)
		g_block[lane] = uint2(block_start, info);

	const uint packed = PDEP4_PACKED[lane];
	[unroll] for (uint e = 0; e < 4; e++)
		g_pdep[lane * 4u + e] = (packed >> (8u * e)) & 0xffu;

	GroupMemoryBarrierWithGroupSync();

	const uint2 block = g_block[blk];
	const uint control = block.y & 0xffffu;
	const uint q_bits = (block.y >> 16u) & 15u;
	// The CDF 9/7 input scaling of the band (see xbox/idwt.hlsl): LL K^2, HH 1/K^2.
	const float K2 = 1.230174104914001f * 1.230174104914001f;
	const float band_scale = output_layer == 0 ? K2 : (output_layer == 3 ? 1.0f / K2 : 1.0f);
	const float half_scale = 0.5f * band_scale * (decode_quant(q_code) * decode_quant_scale((block.y >> 20u) & 15u));

	const uint below = control & ((1u << (2u * sub)) - 1u);
	const uint pos0 = block.x + q_bits * sub + countbits(below & 0x5555u) + 2u * countbits(below & 0xaaaau);
	const uint n0 = q_bits + ((control >> (2u * sub)) & 3u);
	const uint n1 = q_bits + ((control >> (2u * sub + 2u)) & 3u);

	const SubBlock s0 = decode(pos0, n0);
	const SubBlock s1 = decode(pos0 + n0, n1);

	const uint sign0 = sign_base + WavePrefixSum(s0.count + s1.count);
	finish(s0, sign0, half_scale, origin);
	finish(s1, sign0 + s0.count, half_scale, origin + uint2(0, 2));

	GroupMemoryBarrierWithGroupSync();
	store_tile(group.xy, lane, false);
}

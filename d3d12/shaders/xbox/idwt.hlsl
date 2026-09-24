// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
//
// Inverse DWT (CDF 9/7 synthesis) of one decomposition level for the moonlight-xbox
// client on Xbox Series. Not a general-purpose kernel: it is tuned for the GPU
// partition that client runs in (a UWP declaring hevcPlayback, which it needs for
// HDR10 output, gets the console's "4K media app" share: ~8% of the ALU throughput,
// ~116 GB/s, LDS cut like ALU, ~5x latency; see xbox_kernel_model in
// pyrowave_d3d12_device.cpp), where ALU, LDS and memory are all scarce and the pattern
// of each store instruction decides its cost, and for the rates it streams at. There
// the full-resolution level of a 4K 4:4:4 frame at ~460 KB went from 1.7 ms
// (shaders/idwt.comp) to 1.07 ms. On desktop GPUs it is no faster, and the device only
// uses it where waves are always 64 lanes, i.e. on the Xbox.
//
// Same inputs, outputs and boundary handling as shaders/idwt.comp, restructured:
//
// - A group makes a 64x56 output tile (idwt.comp: 32x32 with 64 threads), so the
//   2-sample apron the lifting steps need on every side costs less.
// - Each lane lifts one 32-sample row segment, then one 28-sample column segment,
//   in FP32 registers. idwt.comp lifts 8-sample segments, redoing most of the apron
//   work for every 8 outputs, and rounds to FP16 between the passes.
// - The tile is staged in LDS and written one 8x8 square per store instruction.
// - LDS bandwidth is as scarce as ALU there, so what passes through LDS is packed:
//   the band window is gathered as raw FP16 bits (P1_FP16), the output is staged as
//   the 16-bit values the store writes anyway (STAGE_U16), both exact, and the
//   row-to-column exchange is rounded to FP16 (XCHG_FP16; ~0.27 ms of 1.8 there).
//   idwt.comp rounds to FP16 between the passes too, but truncating; this rounds to
//   nearest even, as it does before every FP16 store (the hardware conversion
//   truncates), and ends up far closer to an FP32 decode than idwt.comp (16-bit PSNR
//   ~79-81 dB vs 64-66 dB at the rates tested).
//
// The lifting's K / 1/K input scaling is not done here: across both passes it
// multiplies LL by K^2, HL and LH by 1, and HH by 1/K^2, so the Xbox dequant stores
// HH (and the coarsest LL) prescaled, and every level but the last writes its output,
// the next level's LL, multiplied by K^2 (LL_SCALE).
//
// For the two finest levels (flag in block_stride's top bit), the Xbox dequant does
// not write the 32x32 blocks of the detail bands it has no coefficients for (at
// moonlight-xbox's streaming rates that is nearly all of them), so their texels are
// stale and must not be read: they are zero by definition. The per-block offset table
// says which blocks are empty. (The coarser levels are small and latency-bound; the
// lookups would cost them more than the zeros do.) A band whose part of the window is
// all empty is neither gathered nor passed through LDS, and when both vertically high
// bands are empty, the odd rows of both passes are skipped.
//
// Band layers: 0 LL, 1 HL (horizontally high), 2 LH (vertically high), 3 HH.
// Symmetric extension at the band edges, as idwt.comp's mirrored sampling gives:
// low bands whole-sample at the start and half-sample at the end, high bands the
// reverse.

#ifndef PW_WAVE_SIZE_ATTRIBUTE
#define PW_WAVE_SIZE_ATTRIBUTE
#endif

// Final level: add the DC offset. Otherwise the output is the next level's LL band and
// is prescaled by K^2.
#ifndef DC_SHIFT
#define DC_SHIFT 0
#endif

#ifndef P1_FP16
#define P1_FP16 1
#endif
#ifndef XCHG_FP16
#define XCHG_FP16 1
#endif
#ifndef STAGE_U16
#define STAGE_U16 1
#endif
// Staging the output tile as 16-bit values: UNORM16 for the final level's planes, FP16
// (already rounded to it) for the next level's LL band.
#define STAGE_PACKED STAGE_U16

cbuffer Registers : register(b0)
{
	int2 band_size;
	float2 inv_band_size;
	// First entry in the block offset table of the HL, LH and HH band, and the band's
	// width in 32x32 blocks | CHECK_EMPTY_BLOCKS.
	uint3 detail_block_offset;
	uint block_stride_flags;
};
static const uint CHECK_EMPTY_BLOCKS = 0x80000000u;
static const bool check_empty = (block_stride_flags & CHECK_EMPTY_BLOCKS) != 0;
static const uint block_stride = block_stride_flags & ~CHECK_EMPTY_BLOCKS;

ByteAddressBuffer Offsets : register(t1);

#if P1_FP16
Texture2DArray<uint> Bands : register(t0); // the FP16 pyramid viewed as R16_UINT
#define TEXEL4 uint4
#else
Texture2DArray<float> Bands : register(t0);
#define TEXEL4 float4
#endif
SamplerState MirrorPoint : register(s0);
RWTexture2DArray<float4> Output : register(u1);

static const uint TileW = 64;
static const uint TileH = 56;
static const uint BandW = TileW / 2 + 4; // 36, including the 2-coefficient apron
static const uint BandH = TileH / 2 + 4; // 32

static const float ALPHA = -1.586134342059924f;
static const float BETA = -0.052980118572961f;
static const float GAMMA = 0.882911075530934f;
static const float DELTA = 0.443506852043971f;
static const float LL_SCALE = 1.230174104914001f * 1.230174104914001f;

// LDS layouts, in dwords. Pitches are chosen so the phases' accesses are free of bank
// conflicts (32 banks; a half-wave of 32 lanes is served at a time), except the phase
// 1 window, which is packed tighter at the cost of some 2-way conflicts: LDS size is
// what limits how many groups a CU runs, and that matters more.
//
// Phase 1, the four bands' 36x32 apron'd window:
//   P1_FP16: [band * W_BAND][row * W_PITCH][col / 2] as packed half pairs,
//   else     [band][row][col] floats.
static const uint W_PITCH = 18;
static const uint W_BAND = 32 * W_PITCH + 2;
// Phase 2, the horizontally transformed rows (tile rows -4 .. 59):
//   XCHG_FP16: [row * X_PITCH][x / 2] as packed half pairs,
//   else       [row * H_PITCH][segment * H_SEGMENT + x % 32] floats.
static const uint X_PITCH = 33;
static const uint H_PITCH = 65;
static const uint H_SEGMENT = 33;
// Phase 3, the output tile:
//   STAGE_PACKED: [row / 2 * O_PITCH][x] as UNORM16 pairs of rows,
//   else          [row * O_PITCH][x] floats.
static const uint O_PITCH = 72;

#if P1_FP16
static const uint LDS_P1 = 4 * W_BAND;
#else
static const uint LDS_P1 = 4 * BandH * BandW;
#endif
#if XCHG_FP16
static const uint LDS_P2 = 64 * X_PITCH;
#else
static const uint LDS_P2 = 64 * H_PITCH;
#endif
#if STAGE_PACKED
static const uint LDS_P3 = (TileH / 2) * O_PITCH;
#else
static const uint LDS_P3 = TileH * O_PITCH;
#endif
static const uint LDS_P12 = LDS_P1 > LDS_P2 ? LDS_P1 : LDS_P2;
groupshared uint g_lds[LDS_P12 > LDS_P3 ? LDS_P12 : LDS_P3];

// Phase 1 window writes for the 2x2 quad at window position 2 * q.
#if P1_FP16
void store_quad(uint4 g, uint band, uint2 q)
{
	const uint i = band * W_BAND + 2u * q.y * W_PITCH + q.x;
	g_lds[i] = g.w | (g.z << 16u);
	g_lds[i + W_PITCH] = g.x | (g.y << 16u);
}
#else
void store_quad(float4 g, uint band, uint2 q)
{
	const uint i = (band * BandH + 2u * q.y) * BandW + 2u * q.x;
	g_lds[i] = asuint(g.w);
	g_lds[i + 1] = asuint(g.z);
	g_lds[i + BandW] = asuint(g.x);
	g_lds[i + BandW + 1] = asuint(g.y);
}
#endif

// Whether 32x32 block `block` of detail band `band` (1..3) is empty. Blocks past the
// band's edge are never sampled; they count as empty.
bool block_empty(uint band, int2 block)
{
	if (any(block < 0) || block.x >= int(block_stride) || block.y >= (band_size.y + 31) / 32)
		return true;
	const uint first = band == 1u ? detail_block_offset.x : (band == 2u ? detail_block_offset.y : detail_block_offset.z);
	return Offsets.Load((first + uint(block.y) * block_stride + uint(block.x)) * 4u) == 0xffffffffu;
}

// The texture's MIRROR addressing, for coordinates within one period.
int mirror(int t, int n)
{
	return t < 0 ? -t - 1 : (t >= n ? 2 * n - 1 - t : t);
}

// Emptiness of the detail bands' blocks around a tile, as a 48-bit mask: bit
// 16 * (band - 1) + 4 * y + x is block w0 - 1 + (x, y), where w0 is the block of the
// window origin. Every block the window's texels map to, mirrored or not, lies in that
// 4x4 neighbourhood: the window is 36 x 32 texels, and a tile only exists while its
// origin is inside the band.
struct EmptyBlocks
{
	uint2 mask;
	int2 w0;

	bool empty(uint band, int2 block)
	{
		const int2 r = block - w0 + 1;
		const uint i = 16u * (band - 1u) + uint(4 * r.y + r.x);
		return ((i < 32u ? mask.x >> i : mask.y >> (i - 32u)) & 1u) != 0;
	}
};

EmptyBlocks find_empty_blocks(int2 band_origin, uint l)
{
	EmptyBlocks e;
	e.w0 = band_origin >> 5;
	bool empty = false;
	if (l < 48u)
		empty = block_empty(1u + l / 16u, e.w0 - 1 + int2(l & 3u, (l >> 2u) & 3u));
	e.mask = WaveActiveBallot(empty).xy;
	return e;
}

// Loads the 2x2 quad at band coordinate c (even) of `band`. idwt.comp's rule: a low
// band's quad starting before 0 and a high band's quad reaching past the end are
// shifted by one before the mirrored sampling. Near the edges a quad can straddle two
// blocks after mirroring, so emptiness is checked per texel.
void load_quad_edge(uint band, int2 c, uint2 q, EmptyBlocks blocks)
{
	const bool2 high = bool2((band & 1u) != 0, band >= 2u);
	c -= int2(and(!high, c < 0));
	c += int2(and(high, c + 1 >= band_size));
	const float2 uv = float2(c + 1) * inv_band_size;
	TEXEL4 g = Bands.GatherRed(MirrorPoint, float3(uv, float(band)));
	if (band != 0 && check_empty)
	{
		// Gather order: w (x0, y0), z (x1, y0), x (x0, y1), y (x1, y1).
		const int2 e0 = int2(mirror(c.x, band_size.x), mirror(c.y, band_size.y)) >> 5;
		const int2 e1 = int2(mirror(c.x + 1, band_size.x), mirror(c.y + 1, band_size.y)) >> 5;
		if (blocks.empty(band, int2(e0.x, e0.y)))
			g.w = 0;
		if (blocks.empty(band, int2(e1.x, e0.y)))
			g.z = 0;
		if (blocks.empty(band, int2(e0.x, e1.y)))
			g.x = 0;
		if (blocks.empty(band, int2(e1.x, e1.y)))
			g.y = 0;
	}
	store_quad(g, band, q);
}

// Away from the edges: uv is the lane's base plus a constant, and a quad lies in one
// block, whose emptiness the caller looked up.
void load_quad_interior(uint band, float2 uv, uint2 q, bool empty)
{
	if (empty)
		store_quad(TEXEL4(0, 0, 0, 0), band, q);
	else
		store_quad(Bands.GatherRed(MirrorPoint, float3(uv, float(band))), band, q);
}

// One 1D synthesis over n = 2 * (count - 2) output samples: e[i] / o[i] are the low /
// high inputs, sample 2i - 4 / 2i - 3 of the segment. Returns outputs in e[2..] and o[2..].
#define LIFT(e, o, count)                                                   \
	{                                                                       \
		[unroll] for (uint i = 1; i < count; i++)                           \
			e[i] -= DELTA * (o[i - 1] + o[i]);                              \
		[unroll] for (uint i = 1; i < count - 1; i++)                       \
			o[i] -= GAMMA * (e[i] + e[i + 1]);                              \
		[unroll] for (uint i = 2; i < count - 1; i++)                       \
			e[i] -= BETA * (o[i - 1] + o[i]);                               \
		[unroll] for (uint i = 2; i < count - 2; i++)                       \
			o[i] -= ALPHA * (e[i] + e[i + 1]);                              \
	}

// FP32 -> FP16 bits, rounding to nearest even. (f32tof16 compiles to a round-toward-
// zero conversion on AMD, which D3D allows.) Rounds the FP32 mantissa to FP16's 10 bits
// first, after which the truncating conversion is exact; normal range only, which
// suffices for wavelet samples.
uint f32tof16_rtne(float v)
{
	uint b = asuint(v);
	b += 0xfffu + ((b >> 13u) & 1u);
	return f32tof16(asfloat(b & ~0x1fffu));
}

float unorm16_to_float(uint u)
{
	return float(u) * (1.0f / 65535.0f);
}

// A staged 16-bit output value back to what the store takes.
float staged_to_float(uint u)
{
#if DC_SHIFT
	return unorm16_to_float(u);
#else
	return f16tof32(u);
#endif
}

[numthreads(128, 1, 1)]
PW_WAVE_SIZE_ATTRIBUTE
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
	// Every LDS access below is a per-lane base plus a constant, so the addressing
	// folds into the instructions' immediate offsets.
	const uint wave = lane >> 6u;
	const uint l = lane & 63u;

	// Phase 1: gather the window. Lanes cover 16x8 quads per instruction; the last
	// two quad columns of each band take one more, mostly idle, instruction.
	const int2 band_origin = int2(group.xy) * int2(TileW / 2, TileH / 2) - 2;
	// Detail bands that are all zero under this tile (interior tiles only; the edge
	// path writes explicit zeros instead).
	bool zero_hl = false, zero_lh = false, zero_hh = false;
	{
		const uint2 q = uint2(lane & 15u, lane >> 4u);
		const uint2 q_right = uint2(16u + (lane & 1u), lane >> 1u);
		if (!check_empty)
		{
			// Every block is written: plain gathers.
			const float2 uv = float2(band_origin + 2 * int2(q) + 1) * inv_band_size;
			const float2 uv_right = float2(band_origin + 2 * int2(q_right) + 1) * inv_band_size;
			const float2 down = float2(0.0f, 16.0f * inv_band_size.y);
			const bool interior = all(band_origin >= 0) && all(band_origin + int2(BandW, BandH) <= band_size);
			[unroll] for (uint band = 0; band < 4; band++)
			{
				if (interior)
				{
					load_quad_interior(band, uv, q, false);
					load_quad_interior(band, uv + down, q + uint2(0, 8), false);
					if (lane < 32)
						load_quad_interior(band, uv_right, q_right, false);
				}
				else
				{
					EmptyBlocks none = { uint2(0, 0), int2(0, 0) };
					load_quad_edge(band, band_origin + 2 * int2(q), q, none);
					load_quad_edge(band, band_origin + 2 * int2(q) + int2(0, 16), q + uint2(0, 8), none);
					if (lane < 32)
						load_quad_edge(band, band_origin + 2 * int2(q_right), q_right, none);
				}
			}
		}
		else if (all(band_origin >= 0) && all(band_origin + int2(BandW, BandH) <= band_size))
		{
			const float2 uv = float2(band_origin + 2 * int2(q) + 1) * inv_band_size;
			const float2 uv_right = float2(band_origin + 2 * int2(q_right) + 1) * inv_band_size;
			const float2 down = float2(0.0f, 16.0f * inv_band_size.y);

			// LL first: it needs no emptiness check, so its gathers are in flight while
			// the block offsets are looked up.
			load_quad_interior(0, uv, q, false);
			load_quad_interior(0, uv + down, q + uint2(0, 8), false);
			if (lane < 32)
				load_quad_interior(0, uv_right, q_right, false);

			// Inside the band the window only touches neighbourhood columns 1..3 and
			// rows 1..2 (mask bits 0x0eee of each band's 16).
			const EmptyBlocks blocks = find_empty_blocks(band_origin, l);
			zero_hl = (blocks.mask.x & 0x0eeeu) == 0x0eeeu;
			zero_lh = ((blocks.mask.x >> 16u) & 0x0eeeu) == 0x0eeeu;
			zero_hh = (blocks.mask.y & 0x0eeeu) == 0x0eeeu;

			const int2 k = (band_origin + 2 * int2(q)) >> 5;
			const int2 k_down = (band_origin + 2 * int2(q) + int2(0, 16)) >> 5;
			const int2 k_right = (band_origin + 2 * int2(q_right)) >> 5;
			[unroll] for (uint band = 1; band < 4; band++)
			{
				if (band == 1 ? zero_hl : (band == 2 ? zero_lh : zero_hh))
					continue;
				load_quad_interior(band, uv, q, blocks.empty(band, k));
				load_quad_interior(band, uv + down, q + uint2(0, 8), blocks.empty(band, k_down));
				if (lane < 32)
					load_quad_interior(band, uv_right, q_right, blocks.empty(band, k_right));
			}
		}
		else
		{
			const EmptyBlocks blocks = find_empty_blocks(band_origin, l);
			[unroll] for (uint band = 0; band < 4; band++)
			{
				load_quad_edge(band, band_origin + 2 * int2(q), q, blocks);
				load_quad_edge(band, band_origin + 2 * int2(q) + int2(0, 16), q + uint2(0, 8), blocks);
				if (lane < 32)
					load_quad_edge(band, band_origin + 2 * int2(q_right), q_right, blocks);
			}
		}
	}
	GroupMemoryBarrierWithGroupSync();

	// Phase 2: horizontal. Lane -> tile row l - 4 (l = 0..63), 32-sample segment `wave`;
	// its low band is LL (even rows) or LH (odd rows), the high band HL or HH.
	const bool odd = (l & 1u) != 0;
	const bool he_zero = odd && zero_lh;
	const bool ho_zero = odd ? zero_hh : zero_hl;
	const bool odd_rows_zero = zero_lh && zero_hh; // then the odd rows are all zero
	float he[20], ho[20];
	[unroll] for (uint i = 0; i < 20; i++)
		he[i] = ho[i] = 0.0f;
	if (!(odd && odd_rows_zero))
	{
		const uint band = (l & 1u) * 2u;
		const uint row = l >> 1u;
#if P1_FP16
		const uint lo = band * W_BAND + row * W_PITCH + 8u * wave;
		if (!he_zero)
		{
			[unroll] for (uint k = 0; k < 10; k++)
			{
				const uint le = g_lds[lo + k];
				he[2 * k] = f16tof32(le);
				he[2 * k + 1] = f16tof32(le >> 16u);
			}
		}
		if (!ho_zero)
		{
			[unroll] for (uint k = 0; k < 10; k++)
			{
				const uint hh = g_lds[lo + W_BAND + k];
				ho[2 * k] = f16tof32(hh);
				ho[2 * k + 1] = f16tof32(hh >> 16u);
			}
		}
#else
		const uint lo = (band * BandH + row) * BandW + 16u * wave;
		[unroll] for (uint i = 0; i < 20; i++)
		{
			if (!he_zero)
				he[i] = asfloat(g_lds[lo + i]);
			if (!ho_zero)
				ho[i] = asfloat(g_lds[lo + BandH * BandW + i]);
		}
#endif
		LIFT(he, ho, 20)
	}
	GroupMemoryBarrierWithGroupSync();
	if (!(odd && odd_rows_zero))
	{
#if XCHG_FP16
		const uint base = l * X_PITCH + 16u * wave - 2u;
		[unroll] for (uint i = 2; i < 18; i++)
			g_lds[base + i] = f32tof16_rtne(he[i]) | (f32tof16_rtne(ho[i]) << 16u);
#else
		const uint base = l * H_PITCH + wave * H_SEGMENT - 4u;
		[unroll] for (uint i = 2; i < 18; i++)
		{
			g_lds[base + 2u * i] = asuint(he[i]);
			g_lds[base + 2u * i + 1u] = asuint(ho[i]);
		}
#endif
	}
	GroupMemoryBarrierWithGroupSync();

	// Phase 3: vertical. Lane -> column l, 28-row segment `wave`.
	float ve[18], vo[18];
	{
#if XCHG_FP16
		const uint base = (28u * wave) * X_PITCH + (l >> 1u);
		const uint shift = (l & 1u) * 16u;
		[unroll] for (uint i = 0; i < 18; i++)
		{
			ve[i] = f16tof32(g_lds[base + (2u * i) * X_PITCH] >> shift);
			vo[i] = odd_rows_zero ? 0.0f : f16tof32(g_lds[base + (2u * i + 1u) * X_PITCH] >> shift);
		}
#else
		const uint base = (28u * wave) * H_PITCH + (l >> 5u) * H_SEGMENT + (l & 31u);
		[unroll] for (uint i = 0; i < 18; i++)
		{
			ve[i] = asfloat(g_lds[base + (2u * i) * H_PITCH]);
			vo[i] = odd_rows_zero ? 0.0f : asfloat(g_lds[base + (2u * i + 1u) * H_PITCH]);
		}
#endif
	}
	LIFT(ve, vo, 18)
	GroupMemoryBarrierWithGroupSync();
	{
#if STAGE_PACKED
		// Row pairs 2i - 4, 2i - 3 of the segment, as the 16-bit values stored.
		const uint base = (14u * wave - 2u) * O_PITCH + l;
		[unroll] for (uint i = 2; i < 16; i++)
		{
#if DC_SHIFT
			const uint ue = uint(round(saturate(ve[i] + 0.5f) * 65535.0f));
			const uint uo = uint(round(saturate(vo[i] + 0.5f) * 65535.0f));
#else
			const uint ue = f32tof16_rtne(ve[i] * LL_SCALE);
			const uint uo = f32tof16_rtne(vo[i] * LL_SCALE);
#endif
			g_lds[base + i * O_PITCH] = ue | (uo << 16u);
		}
#else
		const uint base = (28u * wave - 4u) * O_PITCH + l;
		[unroll] for (uint i = 2; i < 16; i++)
		{
#if DC_SHIFT
			g_lds[base + (2u * i) * O_PITCH] = asuint(ve[i] + 0.5f);
			g_lds[base + (2u * i + 1u) * O_PITCH] = asuint(vo[i] + 0.5f);
#else
			// Rounded to FP16 here, to nearest even, so the store's own conversion to
			// the FP16 pyramid is exact.
			g_lds[base + (2u * i) * O_PITCH] = asuint(f16tof32(f32tof16_rtne(ve[i] * LL_SCALE)));
			g_lds[base + (2u * i + 1u) * O_PITCH] = asuint(f16tof32(f32tof16_rtne(vo[i] * LL_SCALE)));
#endif
		}
#endif
	}
	GroupMemoryBarrierWithGroupSync();

	// Phase 4: stores; wave w takes tile columns 32w .. 32w + 31.
#if STAGE_PACKED
	{
		// Lane -> column 8 * block + (l & 7), row pair 8 * band + (l >> 3); each read
		// yields two rows. 28 row pairs: the last band of 8 is half used.
		const uint2 p = uint2(l & 7u, l >> 3u);
		const int2 origin = int2(group.xy) * int2(TileW, TileH) + int2(32u * wave + p.x, 2u * p.y);
		const uint base = p.y * O_PITCH + 32u * wave + p.x;
		[unroll] for (uint k = 0; k < 16; k++)
		{
			const uint2 sq = uint2(8u * (k & 3u), 8u * (k >> 2u));
			if (sq.y + p.y < TileH / 2)
			{
				const uint pair = g_lds[base + sq.y * O_PITCH + sq.x];
				const int2 xy = origin + int2(sq.x, 2u * sq.y);
				Output[int3(xy, 0)] = float4(staged_to_float(pair & 0xffffu), 0.0f, 0.0f, 0.0f);
				Output[int3(xy + int2(0, 1), 0)] = float4(staged_to_float(pair >> 16u), 0.0f, 0.0f, 0.0f);
			}
		}
	}
#else
	{
		// 8x8 squares: square columns 4w .. 4w + 3, 7 square rows.
		const uint2 p = uint2(l & 7u, l >> 3u);
		const int2 origin = int2(group.xy) * int2(TileW, TileH) + int2(32u * wave + p.x, p.y);
		const uint base = p.y * O_PITCH + 32u * wave + p.x;
		[unroll] for (uint k = 0; k < 28; k++)
		{
			const uint2 sq = uint2(8u * (k & 3u), 8u * (k >> 2u));
			Output[int3(origin + int2(sq), 0)] = float4(asfloat(g_lds[base + sq.y * O_PITCH + sq.x]), 0.0f, 0.0f, 0.0f);
		}
	}
#endif
}

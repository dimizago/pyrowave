// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Micro-benchmarks that characterise a GPU partition (e.g. the reduced resource tier an
// Xbox UWP gets with the hevcPlayback capability): dependent ALU latency, ALU
// throughput, read bandwidth and typed single-channel texture stores.
// Compiled by tier_bench_build.sh into tier_bench_dxil.h.

cbuffer Params : register(b0)
{
	uint iterations;
	uint stride;
	float a;
	float b;
};

RWByteAddressBuffer Result : register(u0);
ByteAddressBuffer Source : register(t0);
RWTexture2D<float> Target : register(u1);
RWTexture2D<float4> Target4 : register(u2);
RWTexture2D<float2> Target2 : register(u3);
Texture2D<float> SrcR16 : register(t1);
Texture2D<float4> SrcRGBA16 : register(t2);
SamplerState PointMirror : register(s0);

// One wave, one long dependent FMA chain: time / (4 * iterations) = FMA latency.
[numthreads(64, 1, 1)]
void alu_latency(uint3 id : SV_DispatchThreadID)
{
	float v = float(id.x) * 1e-3f;
	[loop]
	for (uint i = 0; i < iterations; i++)
	{
		v = mad(v, a, b);
		v = mad(v, a, b);
		v = mad(v, a, b);
		v = mad(v, a, b);
	}
	Result.Store(id.x * 4, asuint(v));
}

// Eight independent chains per thread, many waves: FMA throughput.
[numthreads(256, 1, 1)]
void alu_throughput(uint3 id : SV_DispatchThreadID)
{
	float v0 = float(id.x), v1 = v0 + 1.0f, v2 = v0 + 2.0f, v3 = v0 + 3.0f;
	float v4 = v0 + 4.0f, v5 = v0 + 5.0f, v6 = v0 + 6.0f, v7 = v0 + 7.0f;
	[loop]
	for (uint i = 0; i < iterations; i++)
	{
		v0 = mad(v0, a, b); v1 = mad(v1, a, b); v2 = mad(v2, a, b); v3 = mad(v3, a, b);
		v4 = mad(v4, a, b); v5 = mad(v5, a, b); v6 = mad(v6, a, b); v7 = mad(v7, a, b);
	}
	Result.Store((id.x & 1023) * 4, asuint(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7));
}

// Streaming 16-byte loads: read bandwidth.
[numthreads(256, 1, 1)]
void read_bandwidth(uint3 id : SV_DispatchThreadID)
{
	uint4 acc = 0;
	[loop]
	for (uint i = 0; i < iterations; i++)
		acc ^= Source.Load4((id.x + i * stride) * 16);
	Result.Store4((id.x & 1023) * 16, acc);
}

// One R16_UNORM texel per thread, 8 per thread along x: typed store throughput, the
// way the decoder writes its output planes.
[numthreads(8, 8, 1)]
void typed_store(uint3 id : SV_DispatchThreadID)
{
	[unroll]
	for (uint i = 0; i < 8; i++)
		Target[uint2(id.x * 8 + i, id.y)] = float(i) * a;
}

// Empty dispatch: per-dispatch + barrier overhead.
[numthreads(64, 1, 1)]
void empty_dispatch(uint3 id : SV_DispatchThreadID)
{
	if (id.x == 0xffffffffu)
		Result.Store(0, 1);
}

// Coalesced single-channel stores: lane i writes x = base + i, 8 rows per thread.
[numthreads(64, 1, 1)]
void store_r16_rows(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll]
	for (uint r = 0; r < 8; r++)
		Target[uint2(gid.x * 64 + lane, gid.y * 8 + r)] = float(r) * a;
}

// The iDWT's output pattern: 8x8 unswizzled threads, transposed 2x1 pairs.
[numthreads(64, 1, 1)]
void store_r16_idwt(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	uint y0 = (lane & 1) | (((lane >> 3) & 3) << 1);
	uint x0 = ((lane >> 1) & 3) | (((lane >> 5) & 1) << 2);
	for (uint y = y0; y < 16; y += 8)
		for (uint x = x0; x < 32; x += 8)
		{
			Target[uint2(2 * y + 0, x) + 32 * gid.yx] = float(x) * a;
			Target[uint2(2 * y + 1, x) + 32 * gid.yx] = float(y) * a;
		}
}

// Four channels per texel (RGBA16), coalesced; the texture is a quarter as many texels.
[numthreads(64, 1, 1)]
void store_rgba16_rows(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll]
	for (uint r = 0; r < 8; r++)
		Target4[uint2(gid.x * 64 + lane, gid.y * 8 + r)] = float4(r, r + 1, r + 2, r + 3) * a;
}

// Two channels per texel (RG16), coalesced.
[numthreads(64, 1, 1)]
void store_rg16_rows(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll]
	for (uint r = 0; r < 8; r++)
		Target2[uint2(gid.x * 64 + lane, gid.y * 8 + r)] = float2(r, r + 1) * a;
}

// Raw buffer stores: one dword and four dwords per lane, coalesced.
[numthreads(64, 1, 1)]
void store_raw_dword(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll]
	for (uint r = 0; r < 8; r++)
		Result.Store(((gid.y * 8 + r) * stride + gid.x * 64 + lane) * 4, r);
}

[numthreads(64, 1, 1)]
void store_raw_dword4(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll]
	for (uint r = 0; r < 8; r++)
		Result.Store4(((gid.y * 8 + r) * stride + gid.x * 64 + lane) * 16, uint4(r, r, r, r));
}

// Loads: 2x2 gathers from an R16 texture vs one RGBA16 texel per lane.
[numthreads(64, 1, 1)]
void load_gather_r16(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	float4 acc = 0;
	[unroll]
	for (uint r = 0; r < 8; r++)
	{
		float2 uv = (float2(gid.x * 128 + lane * 2 + 1, (gid.y * 8 + r) * 2 + 1)) / float2(3840, 2160);
		acc += SrcR16.GatherRed(PointMirror, uv);
	}
	if (acc.x == 12345.0f)
		Result.Store(0, asuint(acc.y + acc.z + acc.w));
}

[numthreads(64, 1, 1)]
void load_rgba16(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	float4 acc = 0;
	[unroll]
	for (uint r = 0; r < 8; r++)
		acc += SrcRGBA16.Load(int3(gid.x * 64 + lane, gid.y * 8 + r, 0));
	if (acc.x == 12345.0f)
		Result.Store(0, asuint(acc.y + acc.z + acc.w));
}

// Dequant output layouts: each group writes one 32x32 tile of 16-bit values.
// (a) R16F texture, each lane a 4x4 area.
[numthreads(64, 1, 1)]
void pyr_r16_lane4x4(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	uint2 o = gid.xy * 32 + 4 * uint2(lane & 7, lane >> 3);
	[unroll] for (uint y = 0; y < 4; y++)
		[unroll] for (uint x = 0; x < 4; x++)
			Target[o + uint2(x, y)] = float(x + y) * a;
}

// (b) R16F texture, each store instruction an 8x8 square.
[numthreads(64, 1, 1)]
void pyr_r16_tile8x8(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll] for (uint k = 0; k < 16; k++)
		Target[gid.xy * 32 + uint2((k & 3) * 8 + (lane & 7), (k >> 2) * 8 + (lane >> 3))] = float(k) * a;
}

// (c) RGBA16 texture (4 values per texel), each instruction 8x8 texels (32x8 values).
[numthreads(64, 1, 1)]
void pyr_rgba16_tile(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll] for (uint k = 0; k < 4; k++)
		Target4[uint2(gid.x * 8 + (lane & 7), gid.y * 32 + k * 8 + (lane >> 3))] = float4(k, k, k, k) * a;
}

// (d) Raw buffer, each 32x32 tile a contiguous 2 KB: 8 and 16 bytes per lane.
[numthreads(64, 1, 1)]
void pyr_raw_tiled_x2(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	uint block = gid.y * stride + gid.x;
	[unroll] for (uint k = 0; k < 4; k++)
		Result.Store2(block * 2048 + k * 512 + lane * 8, uint2(k, lane));
}

[numthreads(64, 1, 1)]
void pyr_raw_tiled_x4(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	uint block = gid.y * stride + gid.x;
	[unroll] for (uint k = 0; k < 2; k++)
		Result.Store4(block * 2048 + k * 1024 + lane * 16, uint4(k, lane, k, lane));
}

// (e) Raw buffer, row-major 3840 values per row: each lane 4 values of one row.
[numthreads(64, 1, 1)]
void pyr_raw_rows_x2(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex)
{
	[unroll] for (uint k = 0; k < 4; k++)
	{
		uint row = gid.y * 32 + k * 8 + (lane >> 3);
		Result.Store2(row * 7680 + (gid.x * 32 + (lane & 7) * 4) * 2, uint2(k, lane));
	}
}

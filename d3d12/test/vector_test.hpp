// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// D3D12 decode + compare harness shared by the desktop comparison tool
// (decode_compare.cpp, which produces the reference with the Vulkan library) and the
// Xbox test app (which has no Vulkan and replays test vectors dumped on a PC).
// Does not depend on pyrowave.h.

#include "pyrowave_d3d12.h"

#include <d3d12.h>

#include "../../com_ptr.hpp"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace PyroWaveTest
{
struct Planes
{
	std::vector<uint8_t> data[3];
	// The raw decoder output when the planes are 16-bit (g_output_format R16_UNORM);
	// data[] then holds it rounded to 8 bits for comparison with the reference.
	std::vector<uint16_t> data16[3];
	int width[3] = {};
	int height[3] = {};

	void allocate(int w, int h, bool chroma_444);
};

struct PacketSpan
{
	size_t offset;
	size_t size;
};

// One encoded frame plus the reference decode, as a self-contained file:
//   "PWTV" u32 version, i32 width, i32 height, u32 chroma_444, u32 num_packets,
//   u32 bitstream_size, bitstream bytes, then per packet u32 offset, u32 size,
//   then the three reference planes, tightly packed.
struct TestVector
{
	int width = 0;
	int height = 0;
	bool chroma_444 = false;
	std::vector<uint8_t> bitstream;
	std::vector<PacketSpan> packets;
	Planes reference;
	std::string name;
};

bool write_test_vector(const std::wstring &path, const TestVector &vec);
bool read_test_vector(const std::wstring &path, TestVector &vec);

struct Context
{
	ComPtr<ID3D12Device> device;
	ComPtr<ID3D12CommandQueue> queue;
	ComPtr<ID3D12CommandAllocator> allocator;
	ComPtr<ID3D12GraphicsCommandList> list;
	ComPtr<ID3D12Fence> fence;
	UINT64 fence_value = 0;
	HANDLE event = nullptr;
	std::string adapter_name;

	// Picks the hardware adapter with the most dedicated memory.
	bool init(bool debug, bool gpu_validation);
	~Context();

	void begin();
	UINT64 submit();
	void wait(UINT64 value);
};

struct DecodeStats
{
	double best_ms = 0.0;
	double median_ms = 0.0;
};

// Decodes the frame `iterations` times into R8 planes and reads the last one back.
// `previous`, if set, is decoded first with the same decoder, so any state a frame
// leaves behind (the wavelet pyramid) holds another frame's data, as in a stream.
bool decode(Context &ctx, pyrowave_d3d12_device device, const TestVector &vec, int iterations, Planes &out,
            DecodeStats &stats, std::string &error, const TestVector *previous = nullptr);

struct PlaneDiff
{
	int max_diff = 0;
	size_t mismatches = 0;
	size_t total = 0;
};

PlaneDiff compare(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b);
double psnr(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b);

// Mean SSIM (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5, K1 0.01, K2 0.03,
// over the pixels the window fits around) of two 8-bit planes.
double ssim(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, int width, int height);

// Reads a raw 8-bit planar frame (yuv444p or yuv420p, as allocated in p).
bool read_raw_planes(const char *path, Planes &p);

// Deterministic synthetic frame: gradients, sinusoids of several frequencies, hard
// edges, fine lines and a little noise -- something for every wavelet band.
Planes make_test_image(int width, int height, bool chroma_444);

// Appends printf-style text to a report string.
void appendf(std::string &out, const char *fmt, ...);

// Decodes every vector at wavelet precision 2 (FP32) and 1 (FP16 storage), compares
// against the reference (expected to be a Vulkan PYROWAVE_PRECISION=2 decode) and
// reports GPU decode times. Tolerance is 1 LSB at precision 2 and 2 at precision 1.
std::string run_suite(Context &ctx, const std::vector<TestVector> &vectors, int iterations, bool &pass);

// Milliseconds to sleep between decode iterations (0 = back to back), to mimic a
// stream's one-frame-per-refresh submission pattern.
extern int g_decode_pace_ms;

// Output plane format: DXGI_FORMAT_R8_UNORM (default) or DXGI_FORMAT_R16_UNORM, the
// format the Moonlight client decodes into.
extern DXGI_FORMAT g_output_format;

// Wavelet precisions run_suite covers, in order (default 2 then 1).
extern std::vector<int> g_precisions;

// Bit-exact regression against an earlier build of the decoder: when set, run_suite
// saves (g_baseline_save) or compares the raw output planes of every vector and
// precision as <dir>\<name>_p<precision>.raw.
extern std::wstring g_baseline_dir;
extern bool g_baseline_save;
// Instead of requiring bit-exactness, measure each 16-bit output against the stored
// precision 2 (FP32) output of the same vector and report the error (g_baseline_dir).
extern bool g_baseline_quality;
}

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
bool decode(Context &ctx, pyrowave_d3d12_device device, const TestVector &vec, int iterations, Planes &out,
            DecodeStats &stats, std::string &error);

struct PlaneDiff
{
	int max_diff = 0;
	size_t mismatches = 0;
	size_t total = 0;
};

PlaneDiff compare(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b);
double psnr(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b);

// Deterministic synthetic frame: gradients, sinusoids of several frequencies, hard
// edges, fine lines and a little noise -- something for every wavelet band.
Planes make_test_image(int width, int height, bool chroma_444);

// Appends printf-style text to a report string.
void appendf(std::string &out, const char *fmt, ...);

// Decodes every vector at wavelet precision 2 (FP32) and 1 (FP16 storage), compares
// against the reference (expected to be a Vulkan PYROWAVE_PRECISION=2 decode) and
// reports GPU decode times. Tolerance is 1 LSB at precision 2 and 2 at precision 1.
std::string run_suite(Context &ctx, const std::vector<TestVector> &vectors, int iterations, bool &pass);
}

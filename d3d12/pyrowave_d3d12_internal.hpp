// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// Internal glue for the D3D12 backend: the object behind pyrowave_d3d12_device and
// the root signature layout every PyroWave D3D12 shader is bound through.

#include <d3d12.h>

#include "pyrowave_d3d12.h"
#include "../com_ptr.hpp"
#include "../metal/pyrowave_bitstream.hpp"

#include <stdint.h>

namespace PyroWave
{
namespace D3D12
{
// Threadgroup sizes are baked into the shaders as numthreads.
constexpr uint32_t DequantThreadgroupSize = 128;
constexpr uint32_t IdwtThreadgroupSize = 64;

// One root signature serves every shader. SPIRV-Cross assigns registers from the
// GLSL bindings, so the layout is:
//   b0      root constants (the GLSL push constant block, "cbuffer Registers")
//   t0      idwt: wavelet pyramid, sampled
//   t1..t4  dequant: payload offsets (raw) and payload as R32/R16/R8 typed buffers
//   u0      dequant: wavelet pyramid, written
//   u1      idwt: LL band of the next level, or an output plane
//   s0      mirror-repeat point sampler (static)
// Each register gets its own descriptor table so no table needs null padding.
enum RootParameter : UINT
{
	RootConstants = 0,
	RootTableT0 = 1,
	RootTableT1ToT4 = 2,
	RootTableU0 = 3,
	RootTableU1 = 4,
	RootParameterCount
};
constexpr UINT RootConstantCount = 8;

constexpr int DefaultPrecision = 1;
int requested_precision();
DXGI_FORMAT wavelet_format(int precision);
bool create_compute_pipeline(pyrowave_d3d12_device_opaque *device, ID3D12RootSignature *root_signature,
                             const void *code, size_t size, const char *name, ComPtr<ID3D12PipelineState> &pipeline);
}
}

struct pyrowave_d3d12_device_opaque
{
	ComPtr<ID3D12Device> dev;
	ComPtr<ID3D12RootSignature> root_signature;
	ComPtr<ID3D12PipelineState> dequant_pipeline;
	// Indexed by the DCShift constant.
	ComPtr<ID3D12PipelineState> idwt_pipeline[2];

	pyrowave_d3d12_message_cb message_cb = nullptr;
	void *message_userdata = nullptr;
	int precision = PyroWave::D3D12::DefaultPrecision;
	UINT descriptor_size = 0;

	void log(const char *fmt, ...) const
#if defined(__GNUC__)
			__attribute__((format(printf, 2, 3)))
#endif
			;
};

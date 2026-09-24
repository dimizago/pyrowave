// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// Internal glue for the D3D12 backend: the object behind pyrowave_d3d12_device and
// the root signature layout every PyroWave D3D12 shader is bound through.

#include <d3d12.h>

#include "pyrowave_d3d12.h"
#include "../com_ptr.hpp"
#include "../metal/pyrowave_bitstream.hpp"

#include <mutex>
#include <stdint.h>

namespace PyroWave
{
namespace D3D12
{
// Threadgroup sizes are baked into the shaders as numthreads.
constexpr uint32_t DequantThreadgroupSize = 128;
constexpr uint32_t IdwtThreadgroupSize = 64;
constexpr uint32_t XboxIdwtTileWidth = 64;
constexpr uint32_t XboxIdwtTileHeight = 56;

// One root signature serves every shader. SPIRV-Cross assigns registers from the
// GLSL bindings, so the layout is:
//   b0      root constants (the GLSL push constant block, "cbuffer Registers")
//   t0      idwt: wavelet pyramid, sampled
//   t1..t4  dequant: payload offsets (raw) and payload as R32/R16/R8 typed buffers
//   t5      dequant (Xbox kernel): payload, raw
//   u0      dequant: wavelet pyramid, written
//   u1      idwt: LL band of the next level, or an output plane
//   s0      mirror-repeat point sampler (static)
// Each register gets its own descriptor table so no table needs null padding.
enum RootParameter : UINT
{
	RootConstants = 0,
	RootTableT0 = 1,
	RootTableT1ToT5 = 2,
	RootTableU0 = 3,
	RootTableU1 = 4,
	RootParameterCount
};
constexpr UINT RootConstantCount = 8;

// Encoder root signature (one layout, two variants differing only in the static
// sampler at s0): b0 root constants, t0 the one sampled texture, u0..u5 buffers or
// the DWT's output image. Every storage buffer is a UAV (transpile.py forces it), so
// buffers never change state and stage boundaries are UAV barriers.
enum EncoderRootParameter : UINT
{
	EncRootConstants = 0,
	EncRootTableT0 = 1,
	EncRootTableU0 = 2, // u0..u5 at EncRootTableU0 + n
	EncRootParameterCount = EncRootTableU0 + 6
};
constexpr UINT EncRootConstantCount = 16;

constexpr uint32_t DwtThreadgroupSize = 64;
constexpr uint32_t QuantThreadgroupSize = 128;
constexpr uint32_t AnalyzeThreadgroupSize = 64;
constexpr uint32_t AnalyzeFinalizeThreadgroupSize = 512;
constexpr uint32_t BlockPackingThreadgroupSize = 64;
// resolve_rate_control's workgroup is exactly one wave, pinned to 64 lanes.
constexpr uint32_t ResolveThreadgroupSize = 64;

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
	// The Xbox kernels (shaders/xbox, tuned for moonlight-xbox in the console's
	// hevcPlayback GPU partition; see xbox_kernel_model) are in dequant_pipeline and
	// idwt_pipeline: dequant runs one 64-thread group per 32x32 block instead of 128
	// threads, and the iDWT makes 64x56 tiles with 128 threads (see XboxIdwtTile*).
	bool xbox_kernels = false;
	// Indexed by the DCShift constant.
	ComPtr<ID3D12PipelineState> idwt_pipeline[2];

	// Encoder. Created on demand by the first encoder, so decode-only users do not pay
	// for them. [0] mirror-repeat sampler (DWT), [1] transparent border (quantizer).
	ComPtr<ID3D12RootSignature> encoder_root_signature[2];
	ComPtr<ID3D12PipelineState> dwt_pipeline[2]; // indexed by DCShift
	ComPtr<ID3D12PipelineState> quant_pipeline;
	ComPtr<ID3D12PipelineState> analyze_pipeline;
	ComPtr<ID3D12PipelineState> analyze_finalize_pipeline;
	ComPtr<ID3D12PipelineState> resolve_pipeline;
	ComPtr<ID3D12PipelineState> block_packing_pipeline;
	bool encode_pipelines_ready = false;
	std::mutex encode_pipeline_lock;

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

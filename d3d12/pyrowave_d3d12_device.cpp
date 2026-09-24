// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// D3D12 device object: capability check, root signature and pipelines.

#include "pyrowave_d3d12_internal.hpp"
#include "shaders/generated/pyrowave_dxil.h"

#include <memory>
#include <new>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace PyroWave;
using namespace PyroWave::D3D12;

namespace PyroWave
{
namespace D3D12
{
int requested_precision()
{
	const char *env = getenv("PYROWAVE_PRECISION");
	if (!env)
		return DefaultPrecision;
	// 0 is FP16 math, which needs native 16-bit types; fall back to FP16 storage.
	int precision = atoi(env);
	return precision == 2 ? 2 : 1;
}

DXGI_FORMAT wavelet_format(int precision)
{
	return precision == 2 ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16_FLOAT;
}
}
}

void pyrowave_d3d12_device_opaque::log(const char *fmt, ...) const
{
	char buffer[1024];
	va_list va;
	va_start(va, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, va);
	va_end(va);

	if (message_cb)
	{
		message_cb(message_userdata, buffer);
	}
	else
	{
		fprintf(stderr, "pyrowave-d3d12: %s\n", buffer);
		OutputDebugStringA("pyrowave-d3d12: ");
		OutputDebugStringA(buffer);
		OutputDebugStringA("\n");
	}
}

namespace
{
bool create_root_signature(pyrowave_d3d12_device device)
{
	D3D12_DESCRIPTOR_RANGE ranges[4] = {};
	ranges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
	ranges[1] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 5, 1, 0, 0 };
	ranges[2] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0 };
	ranges[3] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, 0 };

	D3D12_ROOT_PARAMETER params[RootParameterCount] = {};
	params[RootConstants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[RootConstants].Constants.ShaderRegister = 0;
	params[RootConstants].Constants.Num32BitValues = RootConstantCount;
	for (UINT i = 0; i < 4; i++)
	{
		auto &p = params[RootTableT0 + i];
		p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		p.DescriptorTable.NumDescriptorRanges = 1;
		p.DescriptorTable.pDescriptorRanges = &ranges[i];
	}
	for (auto &p : params)
		p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_STATIC_SAMPLER_DESC sampler = {};
	sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
	sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
	sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
	sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
	sampler.MaxLOD = D3D12_FLOAT32_MAX;
	sampler.ShaderRegister = 0;
	sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	D3D12_ROOT_SIGNATURE_DESC desc = {};
	desc.NumParameters = RootParameterCount;
	desc.pParameters = params;
	desc.NumStaticSamplers = 1;
	desc.pStaticSamplers = &sampler;

	ComPtr<ID3DBlob> blob, error;
	if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, reinterpret_cast<ID3DBlob **>(blob.ppv()),
	                                       reinterpret_cast<ID3DBlob **>(error.ppv()))))
	{
		device->log("Failed to serialize root signature: %s",
		            error ? static_cast<const char *>(error->GetBufferPointer()) : "unknown error");
		return false;
	}

	if (FAILED(device->dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
	                                            __uuidof(ID3D12RootSignature), device->root_signature.ppv())))
	{
		device->log("Failed to create root signature.");
		return false;
	}

	return true;
}

bool create_pipeline(pyrowave_d3d12_device device, const void *code, size_t size, const char *name,
                     ComPtr<ID3D12PipelineState> &pipeline)
{
	return create_compute_pipeline(device, device->root_signature.get(), code, size, name, pipeline);
}

// The Xbox decoder kernels (shaders/xbox) exist for one deployment: the
// moonlight-xbox client decoding a PyroWave stream on Xbox Series. That client has to
// declare the hevcPlayback capability to switch the TV to HDR10, which puts it in the
// console's "4K media app" GPU partition whatever the Dev Home app type says. Measured
// there (d3d12/test/tier_bench), the partition gets about 8% of the GPU's ALU
// throughput, about 116 GB/s of memory bandwidth, LDS throughput cut about as hard as
// ALU, and roughly 5x the latency, and the cost of a typed store depends mostly on how
// compact each instruction's 64 texels are. The translated kernels took 4.5 ms for a
// 4K 4:4:4 frame at streaming rates there; these take 1.8 ms (0.98 -> 0.46 ms with the
// full Game-tier GPU).
//
// They are not a general improvement: on a desktop RDNA3 GPU they measured no faster
// (slower at 1080p), and GPUs without 64-lane waves cannot run them. So they are the
// default only where waves are always 64 lanes, which among D3D12 targets means the
// Xbox. Returns which build to use: 64 (Shader Model 6.4, fixed 64-lane waves), 66
// (Shader Model 6.6 with [WaveSize(64)], for testing on a PC GPU, only when
// PYROWAVE_D3D12_XBOX_KERNELS=1), or 0 for the translated kernels
// (PYROWAVE_D3D12_XBOX_KERNELS=0 forces that on Xbox too).
int xbox_kernel_model(ID3D12Device *dev)
{
	const char *env = getenv("PYROWAVE_D3D12_XBOX_KERNELS");
	const bool forced_on = env && env[0] == '1';
	if (env && env[0] == '0')
		return 0;

	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1))) ||
	    !options1.WaveOps)
		return 0;
	if (options1.WaveLaneCountMin == 64 && options1.WaveLaneCountMax == 64)
		return 64;

	// A PC GPU: only on request, to test the Xbox kernels without a console.
	D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_6 };
	if (forced_on && SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
	    sm.HighestShaderModel >= D3D_SHADER_MODEL_6_6 && options1.WaveLaneCountMin <= 64 &&
	    options1.WaveLaneCountMax >= 64)
		return 66;
	return 0;
}
}

bool PyroWave::D3D12::create_compute_pipeline(pyrowave_d3d12_device device, ID3D12RootSignature *root_signature,
                                              const void *code, size_t size, const char *name,
                                              ComPtr<ID3D12PipelineState> &pipeline)
{
	D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
	desc.pRootSignature = root_signature;
	desc.CS = { code, size };
	HRESULT hr = device->dev->CreateComputePipelineState(&desc, __uuidof(ID3D12PipelineState), pipeline.ppv());
	if (FAILED(hr))
	{
		device->log("Failed to create %s pipeline (hr 0x%08lx).", name, static_cast<unsigned long>(hr));
		return false;
	}

	wchar_t wname[64];
	swprintf(wname, 64, L"pyrowave %hs", name);
	pipeline->SetName(wname);
	return true;
}

const char *pyrowave_d3d12_result_to_string(pyrowave_d3d12_result result)
{
	switch (result)
	{
	case PYROWAVE_D3D12_SUCCESS: return "success";
	case PYROWAVE_D3D12_ERROR_GENERIC: return "generic error";
	case PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT: return "invalid argument";
	case PYROWAVE_D3D12_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
	case PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
	case PYROWAVE_D3D12_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
	case PYROWAVE_D3D12_ERROR_CORRUPT_BITSTREAM: return "corrupt bitstream";
	default: return "unknown error";
	}
}

bool pyrowave_d3d12_device_supports_encoder(ID3D12Device *d3d12_device)
{
	if (!pyrowave_d3d12_device_is_supported(d3d12_device))
		return false;

	D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_6 };
	if (FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) ||
	    sm.HighestShaderModel < D3D_SHADER_MODEL_6_6)
		return false;

	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {};
	if (FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1))) ||
	    FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4))))
		return false;

	return options4.Native16BitShaderOpsSupported && options1.WaveLaneCountMin <= 64 &&
	       options1.WaveLaneCountMax >= 64;
}

bool pyrowave_d3d12_device_is_supported(ID3D12Device *d3d12_device)
{
	if (!d3d12_device)
		return false;

	D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_4 };
	if (FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) ||
	    sm.HighestShaderModel < D3D_SHADER_MODEL_6_4)
		return false;

	D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {};
	if (FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1))))
		return false;

	return options1.WaveOps && options1.WaveLaneCountMin >= 4;
}

pyrowave_d3d12_result pyrowave_d3d12_device_create(const pyrowave_d3d12_device_create_info *info,
                                                   pyrowave_d3d12_device *device)
{
	if (!info || !device || !info->d3d12_device)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	auto created = std::unique_ptr<pyrowave_d3d12_device_opaque>(new (std::nothrow) pyrowave_d3d12_device_opaque);
	if (!created)
		return PYROWAVE_D3D12_ERROR_OUT_OF_HOST_MEMORY;

	created->message_cb = info->message_callback;
	created->message_userdata = info->message_userdata;

	if (!pyrowave_d3d12_device_is_supported(info->d3d12_device))
	{
		created->log("Device does not support Shader Model 6.4 with wave operations.");
		return PYROWAVE_D3D12_ERROR_UNSUPPORTED_DEVICE;
	}

	info->d3d12_device->AddRef();
	created->dev = info->d3d12_device;
	if (info->wavelet_precision != 0 && info->wavelet_precision != 1 && info->wavelet_precision != 2)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	created->precision = info->wavelet_precision ? info->wavelet_precision : requested_precision();
	created->descriptor_size =
			created->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

	if (!create_root_signature(created.get()))
		return PYROWAVE_D3D12_ERROR_GENERIC;

	const bool p2 = created->precision == 2;
	// The Xbox iDWT reads the FP16 pyramid as raw bits, so the Xbox kernels are for
	// precision 1, the one moonlight-xbox uses.
	const int xbox_model = created->precision == 1 ? xbox_kernel_model(created->dev.get()) : 0;
	if (xbox_model)
	{
#define PW_XBOX(name) (xbox_model == 64 ? DXIL::name##_sm64 : DXIL::name##_sm66), \
		(xbox_model == 64 ? sizeof(DXIL::name##_sm64) : sizeof(DXIL::name##_sm66))
		created->xbox_kernels =
				create_pipeline(created.get(), PW_XBOX(wavelet_dequant_xbox), "dequant (Xbox)", created->dequant_pipeline) &&
				create_pipeline(created.get(), PW_XBOX(idwt_xbox), "idwt (Xbox)", created->idwt_pipeline[0]) &&
				create_pipeline(created.get(), PW_XBOX(idwt_xbox_dc), "idwt dc (Xbox)", created->idwt_pipeline[1]);
#undef PW_XBOX
	}
	created->log("Decoder kernels: %s.",
	             created->xbox_kernels ? (xbox_model == 64 ? "Xbox (SM 6.4)" : "Xbox, forced on a PC GPU (SM 6.6, WaveSize 64)")
	                                   : "portable");
	if (!created->xbox_kernels &&
	    (!create_pipeline(created.get(), DXIL::wavelet_dequant, sizeof(DXIL::wavelet_dequant), "dequant",
	                      created->dequant_pipeline) ||
	     !create_pipeline(created.get(), p2 ? DXIL::idwt_p2 : DXIL::idwt_p1,
	                      p2 ? sizeof(DXIL::idwt_p2) : sizeof(DXIL::idwt_p1), "idwt", created->idwt_pipeline[0]) ||
	     !create_pipeline(created.get(), p2 ? DXIL::idwt_p2_dc : DXIL::idwt_p1_dc,
	                      p2 ? sizeof(DXIL::idwt_p2_dc) : sizeof(DXIL::idwt_p1_dc), "idwt dc",
	                      created->idwt_pipeline[1])))
		return PYROWAVE_D3D12_ERROR_GENERIC;

	*device = created.release();
	return PYROWAVE_D3D12_SUCCESS;
}

void pyrowave_d3d12_device_destroy(pyrowave_d3d12_device device)
{
	delete device;
}

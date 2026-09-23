// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// D3D12 backend for the PyroWave encoder. Mirrors the compute pipeline of
// pyrowave_encoder.cpp by way of the Metal port (metal/pyrowave_encoder.mm):
// dwt -> quant -> analyze_rdo (+ finalize) -> resolve_rdo -> block_packing on the GPU,
// then packetize on the CPU with the shared bitstream code.
//
// Like the Metal encoder it owns its queue, because the packet queries have to block
// on completion. The results are written to device-local buffers and copied to
// persistently mapped readback buffers at the end of each encode.
//
// Resource states: every buffer is only ever accessed as a UAV (the shaders declare all
// storage buffers as UAVs), promoted from COMMON on first use in each command list, so
// stages are separated by UAV barriers; the result buffers alone are transitioned to
// COPY_SOURCE for the readback copy. The wavelet
// pyramid rests in UNORDERED_ACCESS; during the DWT each level is transitioned to
// NON_PIXEL_SHADER_RESOURCE once written (the next level samples its LL band, the
// quantizer samples everything), and all of it goes back to UNORDERED_ACCESS at the end.

#include "pyrowave_d3d12_internal.hpp"
#include "shaders/generated/pyrowave_dxil.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

using namespace PyroWave;
using namespace PyroWave::D3D12;

namespace
{
// Root constant layouts, matching the cbuffer Registers declarations SPIRV-Cross
// emitted into shaders/hlsl/*.hlsl (tightly packed, no padding).
struct DwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
	int32_t aligned_resolution[2];
};

struct QuantPush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	float inv_resolution[2];
	float input_layer;
	float quant_resolution;
	int32_t block_offset;
	int32_t block_stride;
	float rdo_distortion_scale;
};

struct AnalyzePush
{
	int32_t resolution[2];
	int32_t resolution_8x8_blocks[2];
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	uint32_t total_wg_count;
	uint32_t num_blocks_aligned;
	uint32_t block_index_shamt;
};

struct ResolvePush
{
	uint32_t target_payload_size;
	uint32_t num_blocks_per_subdivision;
};

struct BlockPackingPush
{
	int32_t resolution[2];
	int32_t resolution_32x32_blocks[2];
	int32_t resolution_8x8_blocks[2];
	uint32_t quant_resolution_code;
	uint32_t sequence_code;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
	int32_t block_offset_8x8;
	int32_t block_stride_8x8;
};

static_assert(sizeof(QuantPush) <= EncRootConstantCount * 4, "QuantPush too large.");
static_assert(sizeof(AnalyzePush) <= EncRootConstantCount * 4, "AnalyzePush too large.");
static_assert(sizeof(BlockPackingPush) <= EncRootConstantCount * 4, "BlockPackingPush too large.");

//////
// Initial quantization resolution. Pure float math from pyrowave_encoder.cpp, as in
// the Metal port.

float get_noise_power_normalized_quant_resolution(int level, int component, int band, int precision)
{
	// Flat spectrum with noise power normalization. The low-pass gain for CDF 9/7 is
	// 6 dB (1 bit), and every decomposition level subtracts 6 dB.
	int bits = precision >= 1 ? 8 : 6;

	if (band == 0)
		bits += 2;
	else if (band < 3)
		bits += 1;

	bits += level;

	// Chroma starts at level 1, subtract one bit.
	if (component != 0)
		bits--;

	return float(1 << bits);
}

float get_quant_resolution(int level, int component, int band, int precision)
{
	// FP16 range is limited, and this is more than a good enough initial estimate.
	return std::min<float>(precision >= 1 ? 4096.0f : 512.0f,
	                       get_noise_power_normalized_quant_resolution(level, component, band, precision));
}

float get_quant_rdo_distortion_scale(int level, int component, int band, int precision, ChromaSubsampling chroma)
{
	float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
	float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

	constexpr float dpi = 96.0f;
	constexpr float viewing_distance = 1.0f;
	constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

	float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) * cpd_nyquist *
	            std::exp2(-float(level));

	// Don't allow a situation where we're quantizing LL band hard.
	cpd = std::max(cpd, 8.0f);

	float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

	// Heavily discount chroma quality.
	if (component != 0 && level != DecompositionLevels - 1 && chroma == ChromaSubsampling::Chroma420)
		csf *= 0.6f;

	float resolution = get_noise_power_normalized_quant_resolution(level, component, band, precision);
	float weighted_resolution = csf * resolution;
	return weighted_resolution * weighted_resolution;
}

uint32_t floor_log2(uint32_t v)
{
	uint32_t result = 0;
	while (v > 1)
	{
		v >>= 1;
		result++;
	}
	return result;
}

size_t bucket_buffer_size(const BlockLayout &layout)
{
	size_t size = RDOBucketOffset;
	size += size_t(NumRDOBuckets) * BlockSpaceSubdivision * sizeof(uint32_t);
	size += size_t(NumRDOBuckets) * compute_block_count_per_subdivision(layout.block_count_32x32) *
	        BlockSpaceSubdivision * sizeof(RDOperation);
	return size;
}

template <typename Op>
void for_each_band(const BlockLayout &layout, Op op)
{
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
				op(level, component, band);
		}
	}
}

size_t align_up(size_t v, size_t a)
{
	return (v + a - 1) & ~(a - 1);
}

// Buffers the pipeline shares between stages, in the order of their descriptors.
enum Buffer
{
	BufferBlockStats,
	BufferMeta,
	BufferPayload,
	BufferQuant,
	BufferBuckets,
	BufferBitstreamMeta,
	BufferBitstream,
	BufferCount
};

// Buffers zeroed at the start of every encode (accumulated into, or partially written).
enum ClearView
{
	ClearPayloadCounters, // the first two words: allocation counters
	ClearBuckets,
	ClearQuant,
	ClearBitstream,
	ClearViewCount
};

// Descriptor layout of the encoder's shader-visible heap.
constexpr UINT PyramidDescriptorsPerLevel = 3; // 4-band UAV, 4-band SRV, LL SRV
constexpr UINT PyramidDescriptorCount = NumComponents * DecompositionLevels * PyramidDescriptorsPerLevel;
constexpr UINT BufferDescriptorBase = PyramidDescriptorCount;
constexpr UINT ClearDescriptorBase = BufferDescriptorBase + BufferCount;
constexpr UINT InputDescriptorBase = ClearDescriptorBase + ClearViewCount;
constexpr UINT DescriptorCount = InputDescriptorBase + NumComponents;
}

struct pyrowave_d3d12_encoder_opaque
{
	pyrowave_d3d12_device device = nullptr;
	BlockLayout layout;

	ComPtr<ID3D12CommandQueue> queue;
	ComPtr<ID3D12CommandAllocator> allocator;
	ComPtr<ID3D12GraphicsCommandList> list;
	ComPtr<ID3D12Fence> fence;
	UINT64 fence_value = 0;
	HANDLE wait_event = nullptr;
	bool have_result = false;

	ComPtr<ID3D12Resource> wavelet;
	DXGI_FORMAT wavelet_format = DXGI_FORMAT_UNKNOWN;

	ComPtr<ID3D12Resource> buffers[BufferCount];
	UINT64 buffer_sizes[BufferCount] = {};

	// Persistently mapped copies of the results.
	ComPtr<ID3D12Resource> bitstream_readback;
	ComPtr<ID3D12Resource> meta_readback;
	const uint8_t *bitstream_mapped = nullptr;
	const uint8_t *meta_mapped = nullptr;
	size_t bitstream_result_size = 0;

	ComPtr<ID3D12DescriptorHeap> heap;     // shader visible
	ComPtr<ID3D12DescriptorHeap> cpu_heap; // for ClearUnorderedAccessViewUint
	D3D12_CPU_DESCRIPTOR_HANDLE heap_cpu = {}, cpu_heap_start = {};
	D3D12_GPU_DESCRIPTOR_HANDLE heap_gpu = {};

	// CPU input path: reused textures plus an upload buffer.
	ComPtr<ID3D12Resource> cpu_input[2 + 1];
	pyrowave_d3d12_cpu_buffer_format cpu_input_format = PYROWAVE_D3D12_CPU_BUFFER_FORMAT_INT_MAX;
	ComPtr<ID3D12Resource> cpu_upload;
	UINT64 cpu_upload_size = 0;
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT cpu_footprints[3] = {};
	D3D12_RESOURCE_STATES cpu_input_state = D3D12_RESOURCE_STATE_COPY_DEST;

	ComPtr<ID3D12QueryHeap> timestamps;
	ComPtr<ID3D12Resource> timestamp_readback;
	UINT64 timestamp_frequency = 0;
	// PYROWAVE_D3D12_PROFILE=1: a timestamp after every stage, logged after each encode.
	bool profile = false;
	UINT num_stage_marks = 0;
	const char *stage_names[16] = {};
	bool profile_pending = false;

	uint32_t sequence_count = 0;

	D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const
	{
		return { heap_cpu.ptr + SIZE_T(index) * device->descriptor_size };
	}
	D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const
	{
		return { heap_gpu.ptr + UINT64(index) * device->descriptor_size };
	}
	D3D12_CPU_DESCRIPTOR_HANDLE cpu_only(UINT index) const
	{
		return { cpu_heap_start.ptr + SIZE_T(index) * device->descriptor_size };
	}

	static UINT pyramid_uav(int component, int level) { return (component * DecompositionLevels + level) * 3 + 0; }
	static UINT pyramid_srv(int component, int level) { return (component * DecompositionLevels + level) * 3 + 1; }
	static UINT ll_srv(int component, int level) { return (component * DecompositionLevels + level) * 3 + 2; }
	UINT subresource(int level, int layer) const { return UINT(level) + UINT(layer) * DecompositionLevels; }

	void wait_idle()
	{
		if (fence && fence->GetCompletedValue() < fence_value)
		{
			if (SUCCEEDED(fence->SetEventOnCompletion(fence_value, wait_event)))
				WaitForSingleObject(wait_event, INFINITE);
		}
	}

	~pyrowave_d3d12_encoder_opaque()
	{
		wait_idle();
		if (wait_event)
			CloseHandle(wait_event);
	}
};

namespace
{
D3D12_RESOURCE_DESC buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags)
{
	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = size;
	desc.Height = 1;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	desc.Flags = flags;
	return desc;
}

ComPtr<ID3D12Resource> create_buffer(pyrowave_d3d12_device device, D3D12_HEAP_TYPE type, UINT64 size,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state, const wchar_t *name)
{
	D3D12_HEAP_PROPERTIES heap = {};
	heap.Type = type;
	D3D12_RESOURCE_DESC desc = buffer_desc(size, flags);
	ComPtr<ID3D12Resource> res;
	if (FAILED(device->dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
	                                                __uuidof(ID3D12Resource), res.ppv())))
	{
		device->log("Failed to allocate a %llu byte buffer.", static_cast<unsigned long long>(size));
		return {};
	}
	res->SetName(name);
	return res;
}

bool create_encoder_root_signatures(pyrowave_d3d12_device device)
{
	D3D12_DESCRIPTOR_RANGE ranges[7] = {};
	ranges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
	for (UINT i = 0; i < 6; i++)
		ranges[1 + i] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, i, 0, 0 };

	D3D12_ROOT_PARAMETER params[EncRootParameterCount] = {};
	params[EncRootConstants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
	params[EncRootConstants].Constants.ShaderRegister = 0;
	params[EncRootConstants].Constants.Num32BitValues = EncRootConstantCount;
	for (UINT i = 0; i < 7; i++)
	{
		auto &p = params[EncRootTableT0 + i];
		p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		p.DescriptorTable.NumDescriptorRanges = 1;
		p.DescriptorTable.pDescriptorRanges = &ranges[i];
	}
	for (auto &p : params)
		p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

	for (int variant = 0; variant < 2; variant++)
	{
		// [0] mirror repeat for the DWT, [1] transparent black border for the quantizer,
		// so coefficients past the edge of a band read as zero.
		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
		const auto mode = variant == 0 ? D3D12_TEXTURE_ADDRESS_MODE_MIRROR : D3D12_TEXTURE_ADDRESS_MODE_BORDER;
		sampler.AddressU = sampler.AddressV = sampler.AddressW = mode;
		sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

		D3D12_ROOT_SIGNATURE_DESC desc = {};
		desc.NumParameters = EncRootParameterCount;
		desc.pParameters = params;
		desc.NumStaticSamplers = 1;
		desc.pStaticSamplers = &sampler;

		ComPtr<ID3DBlob> blob, error;
		if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
		                                       reinterpret_cast<ID3DBlob **>(blob.ppv()),
		                                       reinterpret_cast<ID3DBlob **>(error.ppv()))) ||
		    FAILED(device->dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		                                            __uuidof(ID3D12RootSignature),
		                                            device->encoder_root_signature[variant].ppv())))
		{
			device->log("Failed to create the encoder root signature.");
			return false;
		}
	}
	return true;
}

bool ensure_encode_pipelines(pyrowave_d3d12_device device)
{
	std::lock_guard<std::mutex> holder{ device->encode_pipeline_lock };
	if (device->encode_pipelines_ready)
		return true;

	if (!pyrowave_d3d12_device_supports_encoder(device->dev.get()))
	{
		device->log("Device lacks Shader Model 6.6, native 16-bit types or 64-wide waves for the encoder.");
		return false;
	}

	if (!create_encoder_root_signatures(device))
		return false;

	auto *mirror = device->encoder_root_signature[0].get();
	auto *border = device->encoder_root_signature[1].get();
	const bool p2 = device->precision == 2;

	bool ok = create_compute_pipeline(device, mirror, p2 ? DXIL::dwt_p2 : DXIL::dwt_p1,
	                                  p2 ? sizeof(DXIL::dwt_p2) : sizeof(DXIL::dwt_p1), "dwt", device->dwt_pipeline[0]) &&
	          create_compute_pipeline(device, mirror, p2 ? DXIL::dwt_p2_dc : DXIL::dwt_p1_dc,
	                                  p2 ? sizeof(DXIL::dwt_p2_dc) : sizeof(DXIL::dwt_p1_dc), "dwt dc",
	                                  device->dwt_pipeline[1]) &&
	          create_compute_pipeline(device, border, DXIL::wavelet_quant, sizeof(DXIL::wavelet_quant), "quant",
	                                  device->quant_pipeline) &&
	          create_compute_pipeline(device, border, DXIL::analyze_rate_control, sizeof(DXIL::analyze_rate_control),
	                                  "analyze", device->analyze_pipeline) &&
	          create_compute_pipeline(device, border, DXIL::analyze_rate_control_finalize,
	                                  sizeof(DXIL::analyze_rate_control_finalize), "analyze finalize",
	                                  device->analyze_finalize_pipeline) &&
	          create_compute_pipeline(device, border, DXIL::resolve_rate_control, sizeof(DXIL::resolve_rate_control),
	                                  "resolve", device->resolve_pipeline) &&
	          create_compute_pipeline(device, border, DXIL::block_packing, sizeof(DXIL::block_packing),
	                                  "block packing", device->block_packing_pipeline);
	device->encode_pipelines_ready = ok;
	return ok;
}

void create_raw_uav(pyrowave_d3d12_encoder encoder, ID3D12Resource *buffer, UINT64 offset, UINT64 size,
                    D3D12_CPU_DESCRIPTOR_HANDLE handle)
{
	D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
	uav.Format = DXGI_FORMAT_R32_TYPELESS;
	uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav.Buffer.FirstElement = offset / 4;
	uav.Buffer.NumElements = UINT(size / 4);
	uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	encoder->device->dev->CreateUnorderedAccessView(buffer, nullptr, &uav, handle);
}

// Clear views are typed R32_UINT and live in both heaps: ClearUnorderedAccessViewUint
// wants a shader-visible handle and a CPU-only one for the same view.
void create_clear_uav(pyrowave_d3d12_encoder encoder, ID3D12Resource *buffer, UINT64 size, UINT index)
{
	D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
	uav.Format = DXGI_FORMAT_R32_UINT;
	uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav.Buffer.NumElements = UINT(size / 4);
	auto *dev = encoder->device->dev.get();
	dev->CreateUnorderedAccessView(buffer, nullptr, &uav, encoder->cpu(ClearDescriptorBase + index));
	dev->CreateUnorderedAccessView(buffer, nullptr, &uav, encoder->cpu_only(index));
}

bool create_pyramid(pyrowave_d3d12_encoder encoder)
{
	auto *device = encoder->device;
	auto &layout = encoder->layout;
	encoder->wavelet_format = wavelet_format(device->precision);

	D3D12_HEAP_PROPERTIES heap = {};
	heap.Type = D3D12_HEAP_TYPE_DEFAULT;
	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Width = UINT64(layout.aligned_width / 2);
	desc.Height = UINT(layout.aligned_height / 2);
	desc.DepthOrArraySize = UINT16(NumFrequencyBandsPerLevel * NumComponents);
	desc.MipLevels = UINT16(DecompositionLevels);
	desc.Format = encoder->wavelet_format;
	desc.SampleDesc.Count = 1;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
	if (FAILED(device->dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
	                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
	                                                __uuidof(ID3D12Resource), encoder->wavelet.ppv())))
	{
		device->log("Failed to allocate the wavelet pyramid.");
		return false;
	}
	encoder->wavelet->SetName(L"pyrowave encoder wavelet pyramid");

	for (int component = 0; component < NumComponents; component++)
	{
		for (int level = 0; level < DecompositionLevels; level++)
		{
			const UINT first_layer = UINT(NumFrequencyBandsPerLevel * component);

			D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
			uav.Format = encoder->wavelet_format;
			uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
			uav.Texture2DArray.MipSlice = UINT(level);
			uav.Texture2DArray.FirstArraySlice = first_layer;
			uav.Texture2DArray.ArraySize = NumFrequencyBandsPerLevel;
			device->dev->CreateUnorderedAccessView(encoder->wavelet.get(), nullptr, &uav,
			                                       encoder->cpu(encoder->pyramid_uav(component, level)));

			D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
			srv.Format = encoder->wavelet_format;
			srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
			srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srv.Texture2DArray.MostDetailedMip = UINT(level);
			srv.Texture2DArray.MipLevels = 1;
			srv.Texture2DArray.FirstArraySlice = first_layer;
			srv.Texture2DArray.ArraySize = NumFrequencyBandsPerLevel;
			device->dev->CreateShaderResourceView(encoder->wavelet.get(), &srv,
			                                      encoder->cpu(encoder->pyramid_srv(component, level)));

			// LL band only: the next level's DWT input.
			srv.Texture2DArray.ArraySize = 1;
			device->dev->CreateShaderResourceView(encoder->wavelet.get(), &srv,
			                                      encoder->cpu(encoder->ll_srv(component, level)));
		}
	}
	return true;
}

// Sized per encode, since the rate control target can change per frame. The GPU is
// idle whenever this runs (encode_frame waits for the previous frame first).
bool ensure_bitstream(pyrowave_d3d12_encoder encoder, size_t size)
{
	size = align_up(size, 256);
	if (encoder->buffers[BufferBitstream] && encoder->buffer_sizes[BufferBitstream] >= size)
		return true;

	auto *device = encoder->device;
	// Buffers are always created in COMMON and promote to UNORDERED_ACCESS on first use.
	encoder->buffers[BufferBitstream] =
			create_buffer(device, D3D12_HEAP_TYPE_DEFAULT, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
			              D3D12_RESOURCE_STATE_COMMON, L"pyrowave bitstream");
	encoder->bitstream_readback = create_buffer(device, D3D12_HEAP_TYPE_READBACK, size, D3D12_RESOURCE_FLAG_NONE,
	                                            D3D12_RESOURCE_STATE_COPY_DEST, L"pyrowave bitstream readback");
	if (!encoder->buffers[BufferBitstream] || !encoder->bitstream_readback)
		return false;
	encoder->buffer_sizes[BufferBitstream] = size;

	void *mapped = nullptr;
	if (FAILED(encoder->bitstream_readback->Map(0, nullptr, &mapped)))
		return false;
	encoder->bitstream_mapped = static_cast<const uint8_t *>(mapped);

	create_raw_uav(encoder, encoder->buffers[BufferBitstream].get(), 0, size,
	               encoder->cpu(BufferDescriptorBase + BufferBitstream));
	create_clear_uav(encoder, encoder->buffers[BufferBitstream].get(), size, ClearBitstream);
	return true;
}

bool create_encode_resources(pyrowave_d3d12_encoder encoder)
{
	auto *device = encoder->device;
	auto *dev = device->dev.get();
	const auto &layout = encoder->layout;

	D3D12_COMMAND_QUEUE_DESC qd = {};
	qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
	if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), encoder->queue.ppv())) ||
	    FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator),
	                                       encoder->allocator.ppv())) ||
	    FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, encoder->allocator.get(), nullptr,
	                                  __uuidof(ID3D12GraphicsCommandList), encoder->list.ppv())) ||
	    FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), encoder->fence.ppv())))
	{
		device->log("Failed to create the encoder queue.");
		return false;
	}
	encoder->queue->SetName(L"pyrowave encode");
	encoder->list->Close();
	encoder->wait_event = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
	if (!encoder->wait_event)
		return false;

	D3D12_DESCRIPTOR_HEAP_DESC hd = {};
	hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	hd.NumDescriptors = DescriptorCount;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), encoder->heap.ppv())))
		return false;
	hd.NumDescriptors = ClearViewCount;
	hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
	if (FAILED(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), encoder->cpu_heap.ppv())))
		return false;
	encoder->heap_cpu = encoder->heap->GetCPUDescriptorHandleForHeapStart();
	encoder->heap_gpu = encoder->heap->GetGPUDescriptorHandleForHeapStart();
	encoder->cpu_heap_start = encoder->cpu_heap->GetCPUDescriptorHandleForHeapStart();

	if (!create_pyramid(encoder))
		return false;

	// Same sizes as the Vulkan and Metal encoders.
	struct
	{
		Buffer index;
		size_t size;
		const wchar_t *name;
	} const scratch[] = {
		{ BufferBlockStats, size_t(layout.block_count_8x8) * sizeof(BlockStatsBlock), L"pyrowave block stats" },
		{ BufferMeta, size_t(layout.block_count_8x8) * sizeof(BlockMeta), L"pyrowave block meta" },
		// Worst case. The first two words are allocation counters; coefficients start at byte 8.
		{ BufferPayload, size_t(layout.aligned_width) * size_t(layout.aligned_height) * 2, L"pyrowave payload" },
		{ BufferQuant, size_t(layout.block_count_32x32) * sizeof(uint32_t), L"pyrowave quant" },
		{ BufferBuckets, bucket_buffer_size(layout), L"pyrowave rdo buckets" },
		{ BufferBitstreamMeta, size_t(layout.block_count_32x32) * sizeof(BitstreamPacket), L"pyrowave bitstream meta" },
	};

	for (auto &entry : scratch)
	{
		const UINT64 size = align_up(entry.size, 256);
		encoder->buffers[entry.index] =
				create_buffer(device, D3D12_HEAP_TYPE_DEFAULT, size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
				              D3D12_RESOURCE_STATE_COMMON, entry.name);
		if (!encoder->buffers[entry.index])
			return false;
		encoder->buffer_sizes[entry.index] = size;
		create_raw_uav(encoder, encoder->buffers[entry.index].get(), 0, size,
		               encoder->cpu(BufferDescriptorBase + entry.index));
	}

	create_clear_uav(encoder, encoder->buffers[BufferPayload].get(), 8, ClearPayloadCounters);
	create_clear_uav(encoder, encoder->buffers[BufferBuckets].get(), encoder->buffer_sizes[BufferBuckets], ClearBuckets);
	create_clear_uav(encoder, encoder->buffers[BufferQuant].get(), encoder->buffer_sizes[BufferQuant], ClearQuant);

	encoder->meta_readback = create_buffer(device, D3D12_HEAP_TYPE_READBACK, encoder->buffer_sizes[BufferBitstreamMeta],
	                                       D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
	                                       L"pyrowave bitstream meta readback");
	void *mapped = nullptr;
	if (!encoder->meta_readback || FAILED(encoder->meta_readback->Map(0, nullptr, &mapped)))
		return false;
	encoder->meta_mapped = static_cast<const uint8_t *>(mapped);

	// GPU timing is best effort.
	D3D12_QUERY_HEAP_DESC qh = {};
	qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	qh.Count = 16;
	const char *profile_env = getenv("PYROWAVE_D3D12_PROFILE");
	encoder->profile = profile_env && profile_env[0] == '1';
	if (FAILED(dev->CreateQueryHeap(&qh, __uuidof(ID3D12QueryHeap), encoder->timestamps.ppv())) ||
	    !(encoder->timestamp_readback = create_buffer(device, D3D12_HEAP_TYPE_READBACK, 16 * sizeof(UINT64), D3D12_RESOURCE_FLAG_NONE,
	                                                  D3D12_RESOURCE_STATE_COPY_DEST, L"pyrowave timestamps")) ||
	    FAILED(encoder->queue->GetTimestampFrequency(&encoder->timestamp_frequency)))
	{
		encoder->timestamps = nullptr;
	}

	return true;
}

//////
// GPU dispatch, one function per stage.

void set_uav(pyrowave_d3d12_encoder encoder, UINT reg, UINT descriptor)
{
	encoder->list->SetComputeRootDescriptorTable(EncRootTableU0 + reg, encoder->gpu(descriptor));
}

void set_buffer(pyrowave_d3d12_encoder encoder, UINT reg, Buffer buffer)
{
	set_uav(encoder, reg, BufferDescriptorBase + buffer);
}

void uav_barrier(ID3D12GraphicsCommandList *list)
{
	D3D12_RESOURCE_BARRIER b = {};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
	list->ResourceBarrier(1, &b);
}

void transition_pyramid_level(pyrowave_d3d12_encoder encoder, int level, D3D12_RESOURCE_STATES before,
                              D3D12_RESOURCE_STATES after)
{
	D3D12_RESOURCE_BARRIER barriers[NumComponents * NumFrequencyBandsPerLevel] = {};
	for (int layer = 0; layer < NumComponents * NumFrequencyBandsPerLevel; layer++)
	{
		auto &b = barriers[layer];
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = encoder->wavelet.get();
		b.Transition.Subresource = encoder->subresource(level, layer);
		b.Transition.StateBefore = before;
		b.Transition.StateAfter = after;
	}
	encoder->list->ResourceBarrier(UINT(NumComponents * NumFrequencyBandsPerLevel), barriers);
}

void dispatch_dwt(pyrowave_d3d12_encoder encoder)
{
	const auto &layout = encoder->layout;
	auto *list = encoder->list.get();
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	list->SetComputeRootSignature(encoder->device->encoder_root_signature[0].get());

	for (int output_level = 0; output_level < DecompositionLevels; output_level++)
	{
		// Each level transforms the LL band the previous one produced. The transition
		// is both the state change and the write -> read barrier.
		if (output_level != 0)
			transition_pyramid_level(encoder, output_level - 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		DwtPush level_push = {};
		if (output_level == 0)
		{
			// The input planes are only `width` wide, but the transform covers the
			// aligned extent and mirrors the source to fill it.
			level_push.resolution[0] = layout.width;
			level_push.resolution[1] = layout.height;
			level_push.aligned_resolution[0] = layout.aligned_width;
			level_push.aligned_resolution[1] = layout.aligned_height;
		}
		else
		{
			level_push.resolution[0] = layout.level_width(output_level - 1);
			level_push.resolution[1] = layout.level_height(output_level - 1);
			level_push.aligned_resolution[0] = level_push.resolution[0];
			level_push.aligned_resolution[1] = level_push.resolution[1];
		}

		// Under 420, chroma is half resolution and enters the pyramid at level 1.
		const int components = (output_level == 0 && chroma_420) ? 1 : NumComponents;
		for (int component = 0; component < components; component++)
		{
			DwtPush push = level_push;
			UINT input;
			// DCShift centres unorm input on zero, so it applies exactly when the
			// source is an input plane rather than a previous level.
			bool reads_input_plane = output_level == 0;

			if (output_level == 0)
			{
				input = InputDescriptorBase + component;
			}
			else if (chroma_420 && component != 0 && output_level == 1)
			{
				input = InputDescriptorBase + component;
				reads_input_plane = true;
				push.resolution[0] = layout.width / 2;
				push.resolution[1] = layout.height / 2;
				push.aligned_resolution[0] = layout.aligned_width >> output_level;
				push.aligned_resolution[1] = layout.aligned_height >> output_level;
			}
			else
			{
				input = encoder->ll_srv(component, output_level - 1);
			}

			push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
			push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

			list->SetPipelineState(encoder->device->dwt_pipeline[reads_input_plane ? 1 : 0].get());
			list->SetComputeRoot32BitConstants(EncRootConstants, sizeof(push) / 4, &push, 0);
			list->SetComputeRootDescriptorTable(EncRootTableT0, encoder->gpu(input));
			set_uav(encoder, 1, encoder->pyramid_uav(component, output_level));
			// Each threadgroup consumes a 32x32 source tile.
			list->Dispatch(UINT(push.aligned_resolution[0] + 31) / 32, UINT(push.aligned_resolution[1] + 31) / 32, 1);
		}
	}

	// The quantizer samples every level.
	transition_pyramid_level(encoder, DecompositionLevels - 1, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
	                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

void dispatch_quant(pyrowave_d3d12_encoder encoder)
{
	const auto &layout = encoder->layout;
	auto *list = encoder->list.get();
	const int precision = encoder->device->precision;

	list->SetComputeRootSignature(encoder->device->encoder_root_signature[1].get());
	list->SetPipelineState(encoder->device->quant_pipeline.get());
	set_buffer(encoder, 1, BufferMeta);
	set_buffer(encoder, 2, BufferBlockStats);
	set_buffer(encoder, 3, BufferPayload);

	for_each_band(layout, [&](int level, int component, int band) {
		QuantPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);
		push.input_layer = float(band);

		// Quantize against the value the decoder will dequantize with: the decoder
		// only ever sees the 8 bit code.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res));
		push.rdo_distortion_scale =
				get_quant_rdo_distortion_scale(level, component, band, precision, layout.chroma) * (1.0f / 256.0f);
		push.block_offset = layout.block_meta[component][level][band].block_offset_8x8;
		push.block_stride = layout.block_meta[component][level][band].block_stride_8x8;

		list->SetComputeRootDescriptorTable(EncRootTableT0, encoder->gpu(encoder->pyramid_srv(component, level)));
		list->SetComputeRoot32BitConstants(EncRootConstants, sizeof(push) / 4, &push, 0);
		list->Dispatch(UINT(push.resolution[0] + 31) / 32, UINT(push.resolution[1] + 31) / 32, 1);
	});
}

void dispatch_analyze_rdo(pyrowave_d3d12_encoder encoder)
{
	const auto &layout = encoder->layout;
	auto *list = encoder->list.get();
	const int per_subdivision = compute_block_count_per_subdivision(layout.block_count_32x32);

	list->SetPipelineState(encoder->device->analyze_pipeline.get());
	set_buffer(encoder, 0, BufferBuckets);
	set_buffer(encoder, 1, BufferBlockStats);

	for_each_band(layout, [&](int level, int component, int band) {
		AnalyzePush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.total_wg_count = uint32_t(layout.block_count_32x32);
		push.num_blocks_aligned = uint32_t(per_subdivision * BlockSpaceSubdivision);
		push.block_index_shamt = floor_log2(uint32_t(per_subdivision));

		list->SetComputeRoot32BitConstants(EncRootConstants, sizeof(push) / 4, &push, 0);
		list->Dispatch(UINT(push.resolution[0] + 31) / 32, UINT(push.resolution[1] + 31) / 32, 1);
	});

	// The finalize pass prefix sums the buckets the dispatches above filled.
	uav_barrier(list);
	list->SetPipelineState(encoder->device->analyze_finalize_pipeline.get());
	list->Dispatch(1, 1, 1);
}

void dispatch_resolve_rdo(pyrowave_d3d12_encoder encoder, size_t target_size)
{
	const auto &layout = encoder->layout;
	auto *list = encoder->list.get();

	// The sequence header is part of the frame's budget.
	if (target_size >= sizeof(BitstreamSequenceHeader))
		target_size -= sizeof(BitstreamSequenceHeader);

	ResolvePush push = {};
	push.target_payload_size = uint32_t(target_size / sizeof(uint32_t));
	push.num_blocks_per_subdivision = uint32_t(compute_block_count_per_subdivision(layout.block_count_32x32));

	list->SetPipelineState(encoder->device->resolve_pipeline.get());
	set_buffer(encoder, 0, BufferBuckets);
	set_buffer(encoder, 1, BufferQuant);
	list->SetComputeRoot32BitConstants(EncRootConstants, sizeof(push) / 4, &push, 0);
	list->Dispatch(UINT(NumRDOBuckets * BlockSpaceSubdivision), 1, 1);
}

void dispatch_block_packing(pyrowave_d3d12_encoder encoder)
{
	const auto &layout = encoder->layout;
	auto *list = encoder->list.get();
	const int precision = encoder->device->precision;

	list->SetPipelineState(encoder->device->block_packing_pipeline.get());
	set_buffer(encoder, 0, BufferBitstream);
	set_buffer(encoder, 1, BufferBitstreamMeta);
	set_buffer(encoder, 2, BufferMeta);
	set_buffer(encoder, 3, BufferPayload);
	set_buffer(encoder, 4, BufferBlockStats);
	set_buffer(encoder, 5, BufferQuant);

	for_each_band(layout, [&](int level, int component, int band) {
		BlockPackingPush push = {};
		push.resolution[0] = layout.level_width(level);
		push.resolution[1] = layout.level_height(level);
		push.resolution_32x32_blocks[0] = (push.resolution[0] + 31) / 32;
		push.resolution_32x32_blocks[1] = (push.resolution[1] + 31) / 32;
		push.resolution_8x8_blocks[0] = (push.resolution[0] + 7) / 8;
		push.resolution_8x8_blocks[1] = (push.resolution[1] + 7) / 8;

		// The code itself here, not the reciprocal the quantizer scales by.
		const float quant_res = get_quant_resolution(level, component, band, precision);
		push.quant_resolution_code = encode_quant(1.0f / quant_res);
		push.sequence_code = encoder->sequence_count;

		const auto &meta = layout.block_meta[component][level][band];
		push.block_offset_32x32 = meta.block_offset_32x32;
		push.block_stride_32x32 = meta.block_stride_32x32;
		push.block_offset_8x8 = meta.block_offset_8x8;
		push.block_stride_8x8 = meta.block_stride_8x8;

		list->SetComputeRoot32BitConstants(EncRootConstants, sizeof(push) / 4, &push, 0);
		// A threadgroup covers 2x2 of the 32x32 blocks, one per 16 lanes.
		list->Dispatch(UINT(push.resolution_32x32_blocks[0] + 1) / 2, UINT(push.resolution_32x32_blocks[1] + 1) / 2, 1);
	});
}

void clear_buffer(pyrowave_d3d12_encoder encoder, ClearView view, Buffer buffer)
{
	static const UINT zero[4] = {};
	encoder->list->ClearUnorderedAccessViewUint(encoder->gpu(ClearDescriptorBase + view), encoder->cpu_only(view),
	                                            encoder->buffers[buffer].get(), zero, 0, nullptr);
}

void copy_result(pyrowave_d3d12_encoder encoder, Buffer buffer, ID3D12Resource *readback, UINT64 size)
{
	auto *list = encoder->list.get();
	D3D12_RESOURCE_BARRIER b = {};
	b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	b.Transition.pResource = encoder->buffers[buffer].get();
	b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
	b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
	list->ResourceBarrier(1, &b);
	list->CopyBufferRegion(readback, 0, encoder->buffers[buffer].get(), 0, size);
	std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
	list->ResourceBarrier(1, &b);
}

// Waits for the previous encode: descriptors, input textures and results are reused.
void wait_previous(pyrowave_d3d12_encoder encoder)
{
	encoder->wait_idle();
}

// Profiling: timestamp index 0 is the start of the encode, 1 the end, 2.. stage ends.
void mark_stage(pyrowave_d3d12_encoder encoder, const char *name)
{
	if (!encoder->profile || !encoder->timestamps || encoder->num_stage_marks >= 14)
		return;
	encoder->stage_names[encoder->num_stage_marks] = name;
	encoder->list->EndQuery(encoder->timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 + encoder->num_stage_marks++);
}

void log_profile(pyrowave_d3d12_encoder encoder)
{
	if (!encoder->profile_pending)
		return;
	encoder->profile_pending = false;
	UINT64 *ts = nullptr;
	D3D12_RANGE range = { 0, 16 * sizeof(UINT64) };
	if (FAILED(encoder->timestamp_readback->Map(0, &range, reinterpret_cast<void **>(&ts))))
		return;
	char line[512];
	int len = snprintf(line, sizeof(line), "profile total %.3f ms:",
	                   double(ts[1] - ts[0]) * 1000.0 / double(encoder->timestamp_frequency));
	UINT64 prev = ts[0];
	for (UINT i = 0; i < encoder->num_stage_marks && len < int(sizeof(line)); i++)
	{
		len += snprintf(line + len, sizeof(line) - len, " %s %.3f", encoder->stage_names[i],
		                double(ts[2 + i] - prev) * 1000.0 / double(encoder->timestamp_frequency));
		prev = ts[2 + i];
	}
	D3D12_RANGE none = { 0, 0 };
	encoder->timestamp_readback->Unmap(0, &none);
	encoder->device->log("%s", line);
}

pyrowave_d3d12_result encode_frame(pyrowave_d3d12_encoder encoder, const pyrowave_d3d12_rate_control *rate_control,
                                   ID3D12Fence *wait_fence, uint64_t wait_value, ID3D12Fence *signal_fence,
                                   uint64_t signal_value, bool cpu_input_recorded)
{
	// Block packing writes u32 words, so round the budget down like the Vulkan encoder.
	const size_t target_size = rate_control->maximum_bitstream_size & ~size_t(3);
	if (target_size == 0 || target_size > UINT32_MAX)
	{
		if (cpu_input_recorded)
			encoder->list->Close();
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	}

	// Same slack the Vulkan encoder leaves for the packer to overshoot into.
	const size_t meta_size = size_t(encoder->layout.block_count_32x32) * sizeof(BitstreamPacket);
	if (!cpu_input_recorded)
	{
		wait_previous(encoder);
		encoder->allocator->Reset();
		encoder->list->Reset(encoder->allocator.get(), nullptr);
	}
	if (!ensure_bitstream(encoder, target_size + meta_size))
	{
		encoder->list->Close();
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;
	}
	encoder->bitstream_result_size = target_size + meta_size;

	encoder->sequence_count = (encoder->sequence_count + 1) & SequenceCountMask;

	auto *list = encoder->list.get();
	ID3D12DescriptorHeap *heaps[] = { encoder->heap.get() };
	list->SetDescriptorHeaps(1, heaps);
	if (encoder->timestamps)
		list->EndQuery(encoder->timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);

	// Accumulated into, so they have to start from zero. The bitstream is cleared too:
	// a block's payload rarely ends on a word boundary, and the leftover bytes of its
	// final word still go out on the wire.
	clear_buffer(encoder, ClearPayloadCounters, BufferPayload);
	clear_buffer(encoder, ClearBuckets, BufferBuckets);
	clear_buffer(encoder, ClearQuant, BufferQuant);
	clear_buffer(encoder, ClearBitstream, BufferBitstream);
	uav_barrier(list);
	encoder->num_stage_marks = 0;
	mark_stage(encoder, "clear");

	dispatch_dwt(encoder);
	mark_stage(encoder, "dwt");
	dispatch_quant(encoder);
	// Rate control analysis reads the per block statistics and payload sizes.
	uav_barrier(list);
	mark_stage(encoder, "quant");
	dispatch_analyze_rdo(encoder);
	uav_barrier(list);
	mark_stage(encoder, "analyze");
	dispatch_resolve_rdo(encoder, target_size);
	// Packing needs the quant decisions resolve just made.
	uav_barrier(list);
	mark_stage(encoder, "resolve");
	dispatch_block_packing(encoder);
	uav_barrier(list);
	mark_stage(encoder, "packing");

	copy_result(encoder, BufferBitstreamMeta, encoder->meta_readback.get(), encoder->buffer_sizes[BufferBitstreamMeta]);
	copy_result(encoder, BufferBitstream, encoder->bitstream_readback.get(), encoder->bitstream_result_size);
	mark_stage(encoder, "readback");

	for (int level = 0; level < DecompositionLevels; level++)
		transition_pyramid_level(encoder, level, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	if (encoder->timestamps)
	{
		list->EndQuery(encoder->timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
		list->ResolveQueryData(encoder->timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 + encoder->num_stage_marks,
		                       encoder->timestamp_readback.get(), 0);
		encoder->profile_pending = encoder->profile;
	}

	if (FAILED(list->Close()))
	{
		encoder->device->log("Failed to record the encode.");
		return PYROWAVE_D3D12_ERROR_GENERIC;
	}

	if (wait_fence)
		encoder->queue->Wait(wait_fence, wait_value);
	ID3D12CommandList *lists[] = { list };
	encoder->queue->ExecuteCommandLists(1, lists);
	encoder->queue->Signal(encoder->fence.get(), ++encoder->fence_value);
	if (signal_fence)
		encoder->queue->Signal(signal_fence, signal_value);

	encoder->have_result = true;
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result wait_for_result(pyrowave_d3d12_encoder encoder)
{
	if (!encoder->have_result)
	{
		encoder->device->log("No frame has been encoded yet.");
		return PYROWAVE_D3D12_ERROR_GENERIC;
	}

	encoder->wait_idle();
	log_profile(encoder);
	HRESULT removed = encoder->device->dev->GetDeviceRemovedReason();
	if (FAILED(removed))
	{
		encoder->device->log("Device removed during encode (hr 0x%08lx).", static_cast<unsigned long>(removed));
		encoder->have_result = false;
		return PYROWAVE_D3D12_ERROR_GENERIC;
	}
	return PYROWAVE_D3D12_SUCCESS;
}

// Input SRVs are Texture2DArray views (the DWT samples a one-layer array; see
// transpile.py), with the value to transform in red.
void create_plane_srv(pyrowave_d3d12_encoder encoder, ID3D12Resource *resource, DXGI_FORMAT format, UINT plane_slice,
                      UINT source_channel, UINT index)
{
	D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
	srv.Format = format;
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
	srv.Shader4ComponentMapping =
			D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(source_channel, source_channel, source_channel, source_channel);
	srv.Texture2DArray.MipLevels = 1;
	srv.Texture2DArray.ArraySize = 1;
	srv.Texture2DArray.PlaneSlice = plane_slice;
	encoder->device->dev->CreateShaderResourceView(resource, &srv, encoder->cpu(InputDescriptorBase + index));
}

pyrowave_d3d12_result set_gpu_input(pyrowave_d3d12_encoder encoder, ID3D12Resource *const planes[3])
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	if (!planes[0])
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	const D3D12_RESOURCE_DESC desc0 = planes[0]->GetDesc();

	if (!planes[1] && !planes[2])
	{
		// One planar NV12 / P010 texture.
		if (desc0.Format != DXGI_FORMAT_NV12 && desc0.Format != DXGI_FORMAT_P010)
		{
			device->log("A single input plane must be NV12 or P010.");
			return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
		}
		if (!chroma_420 || int(desc0.Width) != layout.width || int(desc0.Height) != layout.height)
		{
			device->log("NV12/P010 input must be %dx%d and the encoder 420.", layout.width, layout.height);
			return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
		}
		const bool p010 = desc0.Format == DXGI_FORMAT_P010;
		const DXGI_FORMAT luma = p010 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
		const DXGI_FORMAT chroma = p010 ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;
		create_plane_srv(encoder, planes[0], luma, 0, 0, 0);
		create_plane_srv(encoder, planes[0], chroma, 1, 0, 1);
		create_plane_srv(encoder, planes[0], chroma, 1, 1, 2);
		return PYROWAVE_D3D12_SUCCESS;
	}

	for (int i = 0; i < 3; i++)
	{
		if (!planes[i])
			return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
		const D3D12_RESOURCE_DESC desc = planes[i]->GetDesc();
		const int w = i == 0 || !chroma_420 ? layout.width : layout.width / 2;
		const int h = i == 0 || !chroma_420 ? layout.height : layout.height / 2;
		if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || int(desc.Width) != w || int(desc.Height) != h)
		{
			device->log("Input plane %d must be a %dx%d 2D texture.", i, w, h);
			return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
		}
		create_plane_srv(encoder, planes[i], desc.Format, 0, 0, UINT(i));
	}
	return PYROWAVE_D3D12_SUCCESS;
}

bool ensure_cpu_input(pyrowave_d3d12_encoder encoder, pyrowave_d3d12_cpu_buffer_format format)
{
	if (encoder->cpu_input_format == format && encoder->cpu_input[0])
		return true;

	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const int cw = chroma_420 ? layout.width / 2 : layout.width;
	const int ch = chroma_420 ? layout.height / 2 : layout.height;
	const bool nv12 = format == PYROWAVE_D3D12_CPU_BUFFER_FORMAT_NV12;
	const int num_planes = nv12 ? 2 : 3;

	for (auto &t : encoder->cpu_input)
		t = nullptr;
	encoder->cpu_input_format = PYROWAVE_D3D12_CPU_BUFFER_FORMAT_INT_MAX;

	D3D12_RESOURCE_DESC descs[3] = {};
	for (int i = 0; i < num_planes; i++)
	{
		auto &d = descs[i];
		d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		d.Width = UINT64(i == 0 ? layout.width : cw);
		d.Height = UINT(i == 0 ? layout.height : ch);
		d.DepthOrArraySize = 1;
		d.MipLevels = 1;
		d.Format = (nv12 && i == 1) ? DXGI_FORMAT_R8G8_UNORM : DXGI_FORMAT_R8_UNORM;
		d.SampleDesc.Count = 1;

		D3D12_HEAP_PROPERTIES heap = {};
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;
		if (FAILED(device->dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
		                                                nullptr, __uuidof(ID3D12Resource), encoder->cpu_input[i].ppv())))
		{
			device->log("Failed to allocate the CPU input textures.");
			return false;
		}
	}
	encoder->cpu_input_state = D3D12_RESOURCE_STATE_COPY_DEST;

	UINT64 total = 0;
	for (int i = 0; i < num_planes; i++)
	{
		UINT64 size = 0;
		device->dev->GetCopyableFootprints(&descs[i], 0, 1, total, &encoder->cpu_footprints[i], nullptr, nullptr, &size);
		total = align_up(encoder->cpu_footprints[i].Offset + size, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
	}
	if (total > encoder->cpu_upload_size)
	{
		encoder->cpu_upload = create_buffer(device, D3D12_HEAP_TYPE_UPLOAD, total, D3D12_RESOURCE_FLAG_NONE,
		                                    D3D12_RESOURCE_STATE_GENERIC_READ, L"pyrowave encoder upload");
		if (!encoder->cpu_upload)
			return false;
		encoder->cpu_upload_size = total;
	}

	create_plane_srv(encoder, encoder->cpu_input[0].get(), DXGI_FORMAT_R8_UNORM, 0, 0, 0);
	if (nv12)
	{
		create_plane_srv(encoder, encoder->cpu_input[1].get(), DXGI_FORMAT_R8G8_UNORM, 0, 0, 1);
		create_plane_srv(encoder, encoder->cpu_input[1].get(), DXGI_FORMAT_R8G8_UNORM, 0, 1, 2);
	}
	else
	{
		create_plane_srv(encoder, encoder->cpu_input[1].get(), DXGI_FORMAT_R8_UNORM, 0, 0, 1);
		create_plane_srv(encoder, encoder->cpu_input[2].get(), DXGI_FORMAT_R8_UNORM, 0, 0, 2);
	}

	encoder->cpu_input_format = format;
	return true;
}

// Validates the caller's buffer, copies it into the upload buffer and records the
// upload into the (reset) command list.
pyrowave_d3d12_result record_cpu_input(pyrowave_d3d12_encoder encoder, const pyrowave_d3d12_cpu_buffer *input)
{
	auto *device = encoder->device;
	const auto &layout = encoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const bool nv12 = input->format == PYROWAVE_D3D12_CPU_BUFFER_FORMAT_NV12;
	const int num_planes = nv12 ? 2 : 3;

	if (input->width != layout.width || input->height != layout.height)
	{
		device->log("Input is %dx%d, but the encoder was created for %dx%d.", input->width, input->height,
		            layout.width, layout.height);
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	}
	if (chroma_420 == (input->format == PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV444P))
	{
		device->log("Input format %d does not match the encoder's chroma subsampling.", int(input->format));
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	}
	for (int plane = 0; plane < num_planes; plane++)
	{
		const int w = plane != 0 && chroma_420 ? layout.width / 2 : layout.width;
		const int h = plane != 0 && chroma_420 ? layout.height / 2 : layout.height;
		const size_t bpp = nv12 && plane == 1 ? 2 : 1;
		if (!input->data[plane] || input->row_stride_in_bytes[plane] < size_t(w) * bpp ||
		    input->row_stride_in_bytes[plane] * size_t(h) > input->plane_size_in_bytes[plane])
		{
			device->log("Input plane %d is NULL or has an inconsistent stride and size.", plane);
			return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
		}
	}

	// The textures and upload buffer are reused, so the previous encode must be done.
	wait_previous(encoder);
	if (!ensure_cpu_input(encoder, input->format))
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;

	uint8_t *mapped = nullptr;
	D3D12_RANGE none = { 0, 0 };
	if (FAILED(encoder->cpu_upload->Map(0, &none, reinterpret_cast<void **>(&mapped))))
		return PYROWAVE_D3D12_ERROR_GENERIC;
	for (int plane = 0; plane < num_planes; plane++)
	{
		const auto &fp = encoder->cpu_footprints[plane];
		const size_t row_bytes = size_t(fp.Footprint.Width) * (nv12 && plane == 1 ? 2 : 1);
		for (UINT y = 0; y < fp.Footprint.Height; y++)
			memcpy(mapped + fp.Offset + size_t(y) * fp.Footprint.RowPitch,
			       static_cast<const uint8_t *>(input->data[plane]) + size_t(y) * input->row_stride_in_bytes[plane],
			       row_bytes);
	}
	encoder->cpu_upload->Unmap(0, nullptr);

	encoder->allocator->Reset();
	encoder->list->Reset(encoder->allocator.get(), nullptr);
	auto *list = encoder->list.get();

	D3D12_RESOURCE_BARRIER barriers[3] = {};
	if (encoder->cpu_input_state != D3D12_RESOURCE_STATE_COPY_DEST)
	{
		for (int plane = 0; plane < num_planes; plane++)
		{
			auto &b = barriers[plane];
			b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			b.Transition.pResource = encoder->cpu_input[plane].get();
			b.Transition.StateBefore = encoder->cpu_input_state;
			b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
		}
		list->ResourceBarrier(UINT(num_planes), barriers);
	}
	for (int plane = 0; plane < num_planes; plane++)
	{
		D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
		dst.pResource = encoder->cpu_input[plane].get();
		dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		src.pResource = encoder->cpu_upload.get();
		src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		src.PlacedFootprint = encoder->cpu_footprints[plane];
		list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
	}
	for (int plane = 0; plane < num_planes; plane++)
	{
		auto &b = barriers[plane];
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = encoder->cpu_input[plane].get();
		b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	}
	list->ResourceBarrier(UINT(num_planes), barriers);
	encoder->cpu_input_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	return PYROWAVE_D3D12_SUCCESS;
}
}

//////
// Public API

pyrowave_d3d12_result pyrowave_d3d12_encoder_create(const pyrowave_d3d12_encoder_create_info *info,
                                                    pyrowave_d3d12_encoder *encoder)
{
	if (!info || !encoder || !info->device)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	if (info->chroma != PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420 && info->chroma != PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	const bool chroma_420 = info->chroma == PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420;
	if (chroma_420 && ((info->width & 1) != 0 || (info->height & 1) != 0))
	{
		info->device->log("420 subsampling requires even dimensions, got %dx%d.", info->width, info->height);
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	}

	// Deferred rather than done in device creation, so decode-only users do not pay.
	if (!ensure_encode_pipelines(info->device))
		return PYROWAVE_D3D12_ERROR_UNSUPPORTED_DEVICE;

	auto created = std::unique_ptr<pyrowave_d3d12_encoder_opaque>(new (std::nothrow) pyrowave_d3d12_encoder_opaque);
	if (!created)
		return PYROWAVE_D3D12_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;
	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	if (!create_encode_resources(created.get()))
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;

	*encoder = created.release();
	return PYROWAVE_D3D12_SUCCESS;
}

void pyrowave_d3d12_encoder_destroy(pyrowave_d3d12_encoder encoder)
{
	delete encoder;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_encode_gpu(pyrowave_d3d12_encoder encoder,
                                                        const pyrowave_d3d12_encoder_gpu_input *input,
                                                        const pyrowave_d3d12_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	// The input descriptors are rewritten below; the previous encode may still read them.
	wait_previous(encoder);
	auto result = set_gpu_input(encoder, input->planes);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;

	return encode_frame(encoder, rate_control, input->wait_fence, input->wait_value, input->signal_fence,
	                    input->signal_value, false);
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_encode_cpu(pyrowave_d3d12_encoder encoder,
                                                        const pyrowave_d3d12_cpu_buffer *input,
                                                        const pyrowave_d3d12_rate_control *rate_control)
{
	if (!encoder || !input || !rate_control)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	if (input->format != PYROWAVE_D3D12_CPU_BUFFER_FORMAT_NV12 &&
	    input->format != PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV420P &&
	    input->format != PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV444P)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	auto result = record_cpu_input(encoder, input);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;

	return encode_frame(encoder, rate_control, nullptr, 0, nullptr, 0, true);
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_compute_num_packets_with_padding(pyrowave_d3d12_encoder encoder,
                                                                              size_t packet_boundary,
                                                                              size_t padding_size, size_t *num_packets)
{
	if (!encoder || !num_packets)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;
	*num_packets = compute_num_packets(encoder->layout, encoder->meta_mapped, packet_boundary, padding_size);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_compute_num_packets(pyrowave_d3d12_encoder encoder,
                                                                 size_t packet_boundary, size_t *num_packets)
{
	return pyrowave_d3d12_encoder_compute_num_packets_with_padding(encoder, packet_boundary, 0, num_packets);
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_compute_num_critical_packets(pyrowave_d3d12_encoder encoder, int bands,
                                                                          size_t packet_boundary, size_t padding_size,
                                                                          size_t *num_packets)
{
	if (!encoder || !num_packets || bands < 0 || bands > 4)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;
	*num_packets = compute_num_critical_packets(encoder->layout, bands, encoder->meta_mapped, packet_boundary,
	                                            padding_size);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_packetize_with_padding(pyrowave_d3d12_encoder encoder,
                                                                    pyrowave_d3d12_packet *packets,
                                                                    size_t packet_boundary, size_t padding_size,
                                                                    size_t *out_packets, void *bitstream, size_t size)
{
	if (!encoder || !packets || !out_packets || !bitstream)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;

	static_assert(sizeof(pyrowave_d3d12_packet) == sizeof(Packet), "pyrowave_d3d12_packet layout mismatch.");
	*out_packets = packetize(encoder->layout, reinterpret_cast<Packet *>(packets), packet_boundary, bitstream, size,
	                         encoder->meta_mapped, encoder->bitstream_mapped, padding_size);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_packetize(pyrowave_d3d12_encoder encoder, pyrowave_d3d12_packet *packets,
                                                       size_t packet_boundary, size_t *out_packets, void *bitstream,
                                                       size_t size)
{
	return pyrowave_d3d12_encoder_packetize_with_padding(encoder, packets, packet_boundary, 0, out_packets, bitstream,
	                                                     size);
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_get_mapped_raw_bitstream(pyrowave_d3d12_encoder encoder,
                                                                      const void **mapped_bitstream,
                                                                      size_t *mapped_bitstream_size,
                                                                      const void **mapped_metadata,
                                                                      size_t *mapped_metadata_size)
{
	if (!encoder || !mapped_bitstream || !mapped_bitstream_size || !mapped_metadata || !mapped_metadata_size)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;
	*mapped_bitstream = encoder->bitstream_mapped;
	*mapped_bitstream_size = encoder->bitstream_result_size;
	*mapped_metadata = encoder->meta_mapped;
	*mapped_metadata_size = size_t(encoder->layout.block_count_32x32) * sizeof(BitstreamPacket);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_get_num_active_blocks(pyrowave_d3d12_encoder encoder, int bands,
                                                                   size_t *num_active_blocks)
{
	if (!encoder || !num_active_blocks || bands < 0 || bands > 4)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	*num_active_blocks = get_num_active_blocks(encoder->layout, bands);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_compute_block_active_words(pyrowave_d3d12_encoder encoder, int bands,
                                                                        uint32_t *words, size_t word_count)
{
	if (!encoder || !words || bands < 0 || bands > 4)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;
	compute_block_active_words(encoder->layout, bands, words, word_count, encoder->meta_mapped);
	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_encoder_get_last_gpu_time(pyrowave_d3d12_encoder encoder, double *milliseconds)
{
	if (!encoder || !milliseconds || !encoder->timestamps)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	auto result = wait_for_result(encoder);
	if (result != PYROWAVE_D3D12_SUCCESS)
		return result;

	UINT64 *ts = nullptr;
	D3D12_RANGE range = { 0, 16 };
	if (FAILED(encoder->timestamp_readback->Map(0, &range, reinterpret_cast<void **>(&ts))))
		return PYROWAVE_D3D12_ERROR_GENERIC;
	*milliseconds = double(ts[1] - ts[0]) * 1000.0 / double(encoder->timestamp_frequency);
	D3D12_RANGE written = { 0, 0 };
	encoder->timestamp_readback->Unmap(0, &written);
	return PYROWAVE_D3D12_SUCCESS;
}

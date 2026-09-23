// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// D3D12 backend for the PyroWave decoder. Mirrors the compute path of
// pyrowave_decoder.cpp by way of the Metal port (metal/pyrowave_decoder.mm), whose
// bitstream layer is shared verbatim.
//
// Resource state model: every subresource of the wavelet pyramid rests in
// UNORDERED_ACCESS between decodes. Dequant writes it as UAV; the iDWT then walks
// the levels from coarsest to finest, transitioning one mip at a time to
// NON_PIXEL_SHADER_RESOURCE right before it is sampled (which also makes the LL band
// the previous level wrote visible), and everything is returned to UNORDERED_ACCESS
// at the end.

#include "pyrowave_d3d12_internal.hpp"

#include <memory>
#include <new>
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
// Root constant layouts. These must match the cbuffer Registers declarations
// SPIRV-Cross emitted into shaders/hlsl/*.hlsl.
struct DequantPush
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
};
static_assert(sizeof(DequantPush) <= RootConstantCount * 4, "DequantPush too large.");

struct IdwtPush
{
	int32_t resolution[2];
	float inv_resolution[2];
};
static_assert(sizeof(IdwtPush) <= RootConstantCount * 4, "IdwtPush too large.");

// How many decodes a caller may have in flight before recording one waits for the
// oldest to complete on the GPU.
constexpr UINT UploadSlotCount = 4;

// Per-slot descriptors, rewritten each time the slot is reused:
//   t1 offsets (raw), t2 payload R32, t3 payload R16, t4 payload R8, then 3 plane UAVs.
constexpr UINT SlotPayloadDescriptors = 4;
constexpr UINT SlotDescriptorCount = SlotPayloadDescriptors + 3;

// The dequant shader can read slightly past the end of the payload, so pad.
constexpr UINT64 PayloadPadding = 16;

struct UploadSlot
{
	ComPtr<ID3D12Resource> buffer;
	UINT64 capacity = 0;
	ComPtr<ID3D12Fence> fence;
	UINT64 fence_value = 0;
};
}

struct pyrowave_d3d12_decoder_opaque
{
	pyrowave_d3d12_device device = nullptr;

	BlockLayout layout;
	BitstreamParser parser;

	ComPtr<ID3D12Resource> wavelet;
	DXGI_FORMAT wavelet_format = DXGI_FORMAT_UNKNOWN;

	ComPtr<ID3D12DescriptorHeap> heap;
	D3D12_CPU_DESCRIPTOR_HANDLE heap_cpu = {};
	D3D12_GPU_DESCRIPTOR_HANDLE heap_gpu = {};
	HANDLE wait_event = nullptr;

	UploadSlot slots[UploadSlotCount];
	UINT next_slot = 0;

	// Static descriptor indices.
	UINT pyramid_uav(int component, int level) const { return (component * DecompositionLevels + level) * 3 + 0; }
	UINT pyramid_srv(int component, int level) const { return (component * DecompositionLevels + level) * 3 + 1; }
	UINT ll_uav(int component, int level) const { return (component * DecompositionLevels + level) * 3 + 2; }
	static constexpr UINT StaticDescriptorCount = NumComponents * DecompositionLevels * 3;
	UINT slot_base(UINT slot) const { return StaticDescriptorCount + slot * SlotDescriptorCount; }

	D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const
	{
		return { heap_cpu.ptr + SIZE_T(index) * device->descriptor_size };
	}

	D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const
	{
		return { heap_gpu.ptr + UINT64(index) * device->descriptor_size };
	}

	UINT subresource(int level, int layer) const
	{
		return UINT(level) + UINT(layer) * DecompositionLevels;
	}

	~pyrowave_d3d12_decoder_opaque()
	{
		for (auto &slot : slots)
			wait_slot(slot);
		if (wait_event)
			CloseHandle(wait_event);
	}

	void wait_slot(UploadSlot &slot)
	{
		if (!slot.fence || slot.fence->GetCompletedValue() >= slot.fence_value)
			return;
		// Returns immediately unless the caller really is UploadSlotCount frames ahead.
		if (SUCCEEDED(slot.fence->SetEventOnCompletion(slot.fence_value, wait_event)))
			WaitForSingleObject(wait_event, INFINITE);
	}
};

namespace
{
bool create_wavelet_pyramid(pyrowave_d3d12_decoder decoder)
{
	auto *device = decoder->device;
	auto &layout = decoder->layout;
	decoder->wavelet_format = wavelet_format(device->precision);

	D3D12_HEAP_PROPERTIES heap = {};
	heap.Type = D3D12_HEAP_TYPE_DEFAULT;

	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	desc.Width = UINT64(layout.aligned_width / 2);
	desc.Height = UINT(layout.aligned_height / 2);
	desc.DepthOrArraySize = UINT16(NumFrequencyBandsPerLevel * NumComponents);
	desc.MipLevels = UINT16(DecompositionLevels);
	desc.Format = decoder->wavelet_format;
	desc.SampleDesc.Count = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

	if (FAILED(device->dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
	                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
	                                                __uuidof(ID3D12Resource), decoder->wavelet.ppv())))
	{
		device->log("Failed to allocate the wavelet pyramid.");
		return false;
	}
	decoder->wavelet->SetName(L"pyrowave wavelet pyramid");

	for (int component = 0; component < NumComponents; component++)
	{
		for (int level = 0; level < DecompositionLevels; level++)
		{
			const UINT first_layer = UINT(NumFrequencyBandsPerLevel * component);

			// All four bands of a component at one level, written by dequant.
			D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
			uav.Format = decoder->wavelet_format;
			uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
			uav.Texture2DArray.MipSlice = UINT(level);
			uav.Texture2DArray.FirstArraySlice = first_layer;
			uav.Texture2DArray.ArraySize = NumFrequencyBandsPerLevel;
			device->dev->CreateUnorderedAccessView(decoder->wavelet.get(), nullptr, &uav,
			                                       decoder->cpu(decoder->pyramid_uav(component, level)));

			// The same, sampled by the iDWT.
			D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
			srv.Format = decoder->wavelet_format;
			srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
			srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
			srv.Texture2DArray.MostDetailedMip = UINT(level);
			srv.Texture2DArray.MipLevels = 1;
			srv.Texture2DArray.FirstArraySlice = first_layer;
			srv.Texture2DArray.ArraySize = NumFrequencyBandsPerLevel;
			device->dev->CreateShaderResourceView(decoder->wavelet.get(), &srv,
			                                      decoder->cpu(decoder->pyramid_srv(component, level)));

			// Band 0 (LL) only, written by the iDWT of the level above.
			uav.Texture2DArray.ArraySize = 1;
			device->dev->CreateUnorderedAccessView(decoder->wavelet.get(), nullptr, &uav,
			                                       decoder->cpu(decoder->ll_uav(component, level)));
		}
	}

	return true;
}

bool ensure_slot_buffer(pyrowave_d3d12_decoder decoder, UploadSlot &slot, UINT64 size)
{
	if (slot.buffer && slot.capacity >= size)
		return true;

	// Overallocate so a steadily sized stream stops reallocating.
	UINT64 allocate = size * 2;
	if (allocate < 64 * 1024)
		allocate = 64 * 1024;

	D3D12_HEAP_PROPERTIES heap = {};
	heap.Type = D3D12_HEAP_TYPE_UPLOAD;

	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = allocate;
	desc.Height = 1;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

	slot.buffer = nullptr;
	slot.capacity = 0;
	if (FAILED(decoder->device->dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
	                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
	                                                         __uuidof(ID3D12Resource), slot.buffer.ppv())))
	{
		decoder->device->log("Failed to allocate a %llu byte upload buffer.", static_cast<unsigned long long>(allocate));
		return false;
	}

	slot.buffer->SetName(L"pyrowave decoder upload");
	slot.capacity = allocate;
	return true;
}

void write_payload_descriptors(pyrowave_d3d12_decoder decoder, UINT slot_index, UINT64 offsets_size,
                               UINT64 payload_offset, UINT64 payload_size)
{
	auto *dev = decoder->device->dev.get();
	auto *buffer = decoder->slots[slot_index].buffer.get();
	const UINT base = decoder->slot_base(slot_index);

	// t1: dequant offsets, raw.
	D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
	srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Format = DXGI_FORMAT_R32_TYPELESS;
	srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
	srv.Buffer.FirstElement = 0;
	srv.Buffer.NumElements = UINT(offsets_size / 4 ? offsets_size / 4 : 1);
	dev->CreateShaderResourceView(buffer, &srv, decoder->cpu(base + 0));

	// t2..t4: the same payload bytes as R32, R16 and R8 texel buffers.
	static const struct { DXGI_FORMAT format; UINT size; } views[3] = {
		{ DXGI_FORMAT_R32_UINT, 4 }, { DXGI_FORMAT_R16_UINT, 2 }, { DXGI_FORMAT_R8_UINT, 1 },
	};
	for (UINT i = 0; i < 3; i++)
	{
		srv.Format = views[i].format;
		srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
		srv.Buffer.FirstElement = payload_offset / views[i].size;
		srv.Buffer.NumElements = UINT(payload_size / views[i].size);
		dev->CreateShaderResourceView(buffer, &srv, decoder->cpu(base + 1 + i));
	}
}

bool write_plane_descriptors(pyrowave_d3d12_decoder decoder, UINT slot_index, ID3D12Resource *const planes[3])
{
	auto &layout = decoder->layout;
	auto *device = decoder->device;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;

	for (int i = 0; i < 3; i++)
	{
		if (!planes[i])
		{
			device->log("Output plane %d is NULL.", i);
			return false;
		}

		const D3D12_RESOURCE_DESC desc = planes[i]->GetDesc();
		const int expected_width = i == 0 || !chroma_420 ? layout.width : layout.width / 2;
		const int expected_height = i == 0 || !chroma_420 ? layout.height : layout.height / 2;

		if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1 ||
		    desc.SampleDesc.Count != 1)
		{
			device->log("Output plane %d must be a single-sampled, non-array 2D texture.", i);
			return false;
		}

		if (int(desc.Width) != expected_width || int(desc.Height) != expected_height)
		{
			device->log("Output plane %d is %llux%u, expected %dx%d.", i,
			            static_cast<unsigned long long>(desc.Width), desc.Height, expected_width, expected_height);
			return false;
		}

		if ((desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) == 0)
		{
			device->log("Output plane %d lacks D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.", i);
			return false;
		}

		// The shader declares a one-layer array so the same UAV type also fits the
		// pyramid's LL slices; an array view of a non-array texture is valid.
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.Format = desc.Format;
		uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
		uav.Texture2DArray.ArraySize = 1;
		device->dev->CreateUnorderedAccessView(planes[i], nullptr, &uav,
		                                       decoder->cpu(decoder->slot_base(slot_index) + SlotPayloadDescriptors + i));
	}

	return true;
}

void record_dequant(pyrowave_d3d12_decoder decoder, ID3D12GraphicsCommandList *cmd, UINT slot_index)
{
	auto &layout = decoder->layout;

	cmd->SetPipelineState(decoder->device->dequant_pipeline.get());
	cmd->SetComputeRootDescriptorTable(RootTableT1ToT4, decoder->gpu(decoder->slot_base(slot_index)));

	// Every dispatch writes a distinct (component, level, band) region of the pyramid
	// and none reads another's output, so no barriers are needed between them.
	for (int level = 0; level < DecompositionLevels; level++)
	{
		for (int component = 0; component < NumComponents; component++)
		{
			// Ignore top-level CbCr when doing 420 subsampling.
			if (level == 0 && component != 0 && layout.chroma == ChromaSubsampling::Chroma420)
				continue;

			cmd->SetComputeRootDescriptorTable(RootTableU0, decoder->gpu(decoder->pyramid_uav(component, level)));

			for (int band = (level == DecompositionLevels - 1 ? 0 : 1); band < 4; band++)
			{
				DequantPush push = {};
				push.resolution[0] = layout.level_width(level);
				push.resolution[1] = layout.level_height(level);
				push.output_layer = band;
				push.block_offset_32x32 = layout.block_meta[component][level][band].block_offset_32x32;
				push.block_stride_32x32 = layout.block_meta[component][level][band].block_stride_32x32;
				cmd->SetComputeRoot32BitConstants(RootConstants, sizeof(push) / 4, &push, 0);
				cmd->Dispatch(UINT(push.resolution[0] + 31) / 32, UINT(push.resolution[1] + 31) / 32, 1);
			}
		}
	}
}

void transition_level(pyrowave_d3d12_decoder decoder, ID3D12GraphicsCommandList *cmd, int level,
                      D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
	D3D12_RESOURCE_BARRIER barriers[NumComponents * NumFrequencyBandsPerLevel] = {};
	for (int layer = 0; layer < NumComponents * NumFrequencyBandsPerLevel; layer++)
	{
		auto &b = barriers[layer];
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = decoder->wavelet.get();
		b.Transition.Subresource = decoder->subresource(level, layer);
		b.Transition.StateBefore = before;
		b.Transition.StateAfter = after;
	}
	cmd->ResourceBarrier(UINT(NumComponents * NumFrequencyBandsPerLevel), barriers);
}

void record_idwt_dispatch(pyrowave_d3d12_decoder decoder, ID3D12GraphicsCommandList *cmd, const IdwtPush &push,
                          UINT input_srv, UINT output_uav, bool dc_shift)
{
	cmd->SetPipelineState(decoder->device->idwt_pipeline[dc_shift ? 1 : 0].get());
	cmd->SetComputeRoot32BitConstants(RootConstants, sizeof(push) / 4, &push, 0);
	cmd->SetComputeRootDescriptorTable(RootTableT0, decoder->gpu(input_srv));
	cmd->SetComputeRootDescriptorTable(RootTableU1, decoder->gpu(output_uav));
	cmd->Dispatch(UINT(push.resolution[0] + 15) / 16, UINT(push.resolution[1] + 15) / 16, 1);
}

void record_idwt(pyrowave_d3d12_decoder decoder, ID3D12GraphicsCommandList *cmd, UINT slot_index)
{
	auto &layout = decoder->layout;
	const bool chroma_420 = layout.chroma == ChromaSubsampling::Chroma420;
	const UINT plane_base = decoder->slot_base(slot_index) + SlotPayloadDescriptors;

	for (int input_level = DecompositionLevels - 1; input_level >= 0; input_level--)
	{
		// Levels are a dependent chain: this one samples the LL band the previous one
		// wrote. The transition is both the state change and the write->read barrier.
		transition_level(decoder, cmd, input_level, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		IdwtPush push = {};
		// The shader transposes on load, so resolution is swapped here.
		push.resolution[0] = layout.level_height(input_level);
		push.resolution[1] = layout.level_width(input_level);
		push.inv_resolution[0] = 1.0f / float(push.resolution[0]);
		push.inv_resolution[1] = 1.0f / float(push.resolution[1]);

		if (input_level == 0)
		{
			// Final level writes the output planes directly. Under 420 the chroma
			// planes were already finished one level earlier.
			const int components = chroma_420 ? 1 : NumComponents;
			for (int c = 0; c < components; c++)
				record_idwt_dispatch(decoder, cmd, push, decoder->pyramid_srv(c, input_level), plane_base + c, true);
		}
		else
		{
			for (int c = 0; c < NumComponents; c++)
			{
				const bool final_chroma = chroma_420 && c != 0 && input_level == 1;
				const UINT output = final_chroma ? plane_base + c : decoder->ll_uav(c, input_level - 1);
				record_idwt_dispatch(decoder, cmd, push, decoder->pyramid_srv(c, input_level), output, final_chroma);
			}
		}
	}

	// Back to the resting state for the next frame's dequant.
	for (int level = 0; level < DecompositionLevels; level++)
		transition_level(decoder, cmd, level, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
}

//////
// Public API

pyrowave_d3d12_result pyrowave_d3d12_decoder_create(const pyrowave_d3d12_decoder_create_info *info,
                                                    pyrowave_d3d12_decoder *decoder)
{
	if (!info || !decoder || !info->device)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	if (info->chroma != PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420 &&
	    info->chroma != PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	const bool chroma_420 = info->chroma == PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420;
	if (chroma_420 && ((info->width & 1) != 0 || (info->height & 1) != 0))
	{
		info->device->log("420 subsampling requires even dimensions, got %dx%d.", info->width, info->height);
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	}

	auto created = std::unique_ptr<pyrowave_d3d12_decoder_opaque>(new (std::nothrow) pyrowave_d3d12_decoder_opaque);
	if (!created)
		return PYROWAVE_D3D12_ERROR_OUT_OF_HOST_MEMORY;

	created->device = info->device;
	if (!created->layout.init(info->width, info->height,
	                          chroma_420 ? ChromaSubsampling::Chroma420 : ChromaSubsampling::Chroma444))
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;
	created->parser.init(&created->layout);

	created->wait_event = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
	if (!created->wait_event)
		return PYROWAVE_D3D12_ERROR_GENERIC;

	D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
	heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	heap_desc.NumDescriptors = pyrowave_d3d12_decoder_opaque::StaticDescriptorCount + UploadSlotCount * SlotDescriptorCount;
	heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	if (FAILED(created->device->dev->CreateDescriptorHeap(&heap_desc, __uuidof(ID3D12DescriptorHeap),
	                                                      created->heap.ppv())))
	{
		created->device->log("Failed to create the descriptor heap.");
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;
	}
	created->heap_cpu = created->heap->GetCPUDescriptorHandleForHeapStart();
	created->heap_gpu = created->heap->GetGPUDescriptorHandleForHeapStart();

	if (!create_wavelet_pyramid(created.get()))
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;

	*decoder = created.release();
	return PYROWAVE_D3D12_SUCCESS;
}

void pyrowave_d3d12_decoder_destroy(pyrowave_d3d12_decoder decoder)
{
	delete decoder;
}

void pyrowave_d3d12_decoder_clear(pyrowave_d3d12_decoder decoder)
{
	if (decoder)
		decoder->parser.clear();
}

pyrowave_d3d12_result pyrowave_d3d12_decoder_push_packet(pyrowave_d3d12_decoder decoder, const void *data, size_t size)
{
	if (!decoder || (!data && size != 0))
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	if (!decoder->parser.push_packet(data, size))
		return PYROWAVE_D3D12_ERROR_CORRUPT_BITSTREAM;

	return PYROWAVE_D3D12_SUCCESS;
}

pyrowave_d3d12_result pyrowave_d3d12_decoder_push_packet_truncated(pyrowave_d3d12_decoder decoder, const void *data,
                                                                   size_t size)
{
	if (!decoder || (!data && size != 0))
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	if (!decoder->parser.push_packet(data, size, true))
		return PYROWAVE_D3D12_ERROR_CORRUPT_BITSTREAM;

	return PYROWAVE_D3D12_SUCCESS;
}

bool pyrowave_d3d12_decoder_decode_is_ready(pyrowave_d3d12_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready(allow_partial_frame);
}

bool pyrowave_d3d12_decoder_decode_is_ready_prefix(pyrowave_d3d12_decoder decoder, bool allow_partial_frame)
{
	return decoder && decoder->parser.decode_is_ready_prefix(allow_partial_frame);
}

void pyrowave_d3d12_decoder_get_block_counts(pyrowave_d3d12_decoder decoder, int *decoded_blocks, int *total_blocks)
{
	if (decoded_blocks)
		*decoded_blocks = decoder ? decoder->parser.get_decoded_blocks() : 0;
	if (total_blocks)
		*total_blocks = decoder ? decoder->parser.get_total_blocks_in_sequence() : 0;
}

bool pyrowave_d3d12_decoder_decode_is_ready_with_sideband(pyrowave_d3d12_decoder decoder, bool allow_partial_frame,
                                                          int num_pristine_bands, float minimum_packet_ratio,
                                                          const uint32_t *active_block_mask, size_t word_count)
{
	if (!decoder)
		return false;
	return decoder->parser.decode_is_ready(allow_partial_frame, num_pristine_bands, minimum_packet_ratio,
	                                       active_block_mask, word_count);
}

pyrowave_d3d12_result pyrowave_d3d12_decoder_decode_gpu_buffer(pyrowave_d3d12_decoder decoder,
                                                               ID3D12GraphicsCommandList *command_list,
                                                               const pyrowave_d3d12_gpu_buffers *buffers,
                                                               ID3D12Fence *completion_fence,
                                                               uint64_t completion_value)
{
	if (!decoder || !command_list || !buffers || !completion_fence)
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	const UINT slot_index = decoder->next_slot;
	auto &slot = decoder->slots[slot_index];
	// Applies back pressure rather than growing the ring.
	decoder->wait_slot(slot);

	const auto &offsets = decoder->parser.dequant_offsets();
	const auto &payload = decoder->parser.payload();
	const UINT64 offsets_size = offsets.size() * sizeof(uint32_t);
	const UINT64 payload_offset = (offsets_size + 255) & ~UINT64(255);
	const UINT64 payload_size = payload.size() * sizeof(uint32_t) + PayloadPadding;

	if (!ensure_slot_buffer(decoder, slot, payload_offset + payload_size))
		return PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY;
	if (!write_plane_descriptors(decoder, slot_index, buffers->planes))
		return PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT;

	uint8_t *mapped = nullptr;
	D3D12_RANGE no_read = { 0, 0 };
	if (FAILED(slot.buffer->Map(0, &no_read, reinterpret_cast<void **>(&mapped))))
		return PYROWAVE_D3D12_ERROR_GENERIC;
	if (offsets_size)
		memcpy(mapped, offsets.data(), offsets_size);
	if (!payload.empty())
		memcpy(mapped + payload_offset, payload.data(), payload.size() * sizeof(uint32_t));
	memset(mapped + payload_offset + payload.size() * sizeof(uint32_t), 0, PayloadPadding);
	slot.buffer->Unmap(0, nullptr);

	write_payload_descriptors(decoder, slot_index, offsets_size, payload_offset, payload_size);

	ID3D12DescriptorHeap *heaps[] = { decoder->heap.get() };
	command_list->SetDescriptorHeaps(1, heaps);
	command_list->SetComputeRootSignature(decoder->device->root_signature.get());

	record_dequant(decoder, command_list, slot_index);
	record_idwt(decoder, command_list, slot_index);

	D3D12_RESOURCE_BARRIER uav_barriers[3] = {};
	for (int i = 0; i < 3; i++)
	{
		uav_barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		uav_barriers[i].UAV.pResource = buffers->planes[i];
	}
	command_list->ResourceBarrier(3, uav_barriers);

	// Held until this slot comes round again.
	completion_fence->AddRef();
	slot.fence = completion_fence;
	slot.fence_value = completion_value;
	decoder->next_slot = (slot_index + 1) % UploadSlotCount;

	decoder->parser.mark_frame_decoded();
	return PYROWAVE_D3D12_SUCCESS;
}

// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "tier_bench.hpp"
#include "tier_bench_dxil.h"

#include <algorithm>
#include <functional>
#include <string.h>
#include <vector>

namespace PyroWaveTest
{
namespace
{
struct Bench
{
	Context &ctx;
	ComPtr<ID3D12RootSignature> root;
	ComPtr<ID3D12Resource> result, source, target, target4, target2, src_r16, src_rgba16;
	ComPtr<ID3D12DescriptorHeap> heap;
	ComPtr<ID3D12QueryHeap> queries;
	ComPtr<ID3D12Resource> readback;
	UINT64 frequency = 1;

	explicit Bench(Context &c) : ctx(c) {}

	ComPtr<ID3D12Resource> create(D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC &desc, D3D12_RESOURCE_STATES state)
	{
		D3D12_HEAP_PROPERTIES heap_props = {};
		heap_props.Type = type;
		ComPtr<ID3D12Resource> res;
		if (FAILED(ctx.device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
		                                               __uuidof(ID3D12Resource), res.ppv())))
			return {};
		return res;
	}

	static D3D12_RESOURCE_DESC buffer(UINT64 size, bool uav)
	{
		D3D12_RESOURCE_DESC desc = {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		desc.Width = size;
		desc.Height = 1;
		desc.DepthOrArraySize = 1;
		desc.MipLevels = 1;
		desc.SampleDesc.Count = 1;
		desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
		return desc;
	}

	bool init()
	{
		// Table: u1..u3 (R16, RGBA16, RG16 targets), then t1..t2 (R16F, RGBA16F sources).
		D3D12_DESCRIPTOR_RANGE ranges[2] = {
			{ D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 1, 0, 0 },
			{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1, 0, 3 },
		};
		D3D12_ROOT_PARAMETER params[4] = {};
		params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
		params[0].Constants.Num32BitValues = 4;
		params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
		params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
		params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		params[3].DescriptorTable.NumDescriptorRanges = 2;
		params[3].DescriptorTable.pDescriptorRanges = ranges;
		D3D12_STATIC_SAMPLER_DESC sampler = {};
		sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
		sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
		sampler.MaxLOD = D3D12_FLOAT32_MAX;
		D3D12_ROOT_SIGNATURE_DESC desc = {};
		desc.NumParameters = 4;
		desc.pParameters = params;
		desc.NumStaticSamplers = 1;
		desc.pStaticSamplers = &sampler;
		ComPtr<ID3DBlob> blob, error;
		if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
		                                       reinterpret_cast<ID3DBlob **>(blob.ppv()),
		                                       reinterpret_cast<ID3DBlob **>(error.ppv()))) ||
		    FAILED(ctx.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
		                                           __uuidof(ID3D12RootSignature), root.ppv())))
			return false;

		// Big enough for the raw store tests (3840x2160 dwords, 960x2160 dwordx4).
		result = create(D3D12_HEAP_TYPE_DEFAULT, buffer(3840ull * 2160 * 4, true), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		source = create(D3D12_HEAP_TYPE_DEFAULT, buffer(256ull << 20, false), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

		D3D12_RESOURCE_DESC tex = {};
		tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		tex.Width = 3840;
		tex.Height = 2160;
		tex.DepthOrArraySize = 1;
		tex.MipLevels = 1;
		tex.Format = DXGI_FORMAT_R16_UNORM;
		tex.SampleDesc.Count = 1;
		tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		target = create(D3D12_HEAP_TYPE_DEFAULT, tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		tex.Width = 960;
		tex.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
		target4 = create(D3D12_HEAP_TYPE_DEFAULT, tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		tex.Width = 1920;
		tex.Format = DXGI_FORMAT_R16G16_UNORM;
		target2 = create(D3D12_HEAP_TYPE_DEFAULT, tex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		tex.Flags = D3D12_RESOURCE_FLAG_NONE;
		tex.Width = 3840;
		tex.Format = DXGI_FORMAT_R16_FLOAT;
		src_r16 = create(D3D12_HEAP_TYPE_DEFAULT, tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		tex.Width = 960;
		tex.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
		src_rgba16 = create(D3D12_HEAP_TYPE_DEFAULT, tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
		if (!result || !source || !target || !target4 || !target2 || !src_r16 || !src_rgba16)
			return false;

		D3D12_DESCRIPTOR_HEAP_DESC hd = {};
		hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
		hd.NumDescriptors = 5;
		hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
		if (FAILED(ctx.device->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), heap.ppv())))
			return false;
		D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
		uav.Format = DXGI_FORMAT_R16_UNORM;
		uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		const UINT inc = ctx.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
		ctx.device->CreateUnorderedAccessView(target.get(), nullptr, &uav, h);
		h.ptr += inc;
		uav.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
		ctx.device->CreateUnorderedAccessView(target4.get(), nullptr, &uav, h);
		h.ptr += inc;
		uav.Format = DXGI_FORMAT_R16G16_UNORM;
		ctx.device->CreateUnorderedAccessView(target2.get(), nullptr, &uav, h);
		h.ptr += inc;
		ctx.device->CreateShaderResourceView(src_r16.get(), nullptr, h);
		h.ptr += inc;
		ctx.device->CreateShaderResourceView(src_rgba16.get(), nullptr, h);

		D3D12_QUERY_HEAP_DESC qh = {};
		qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
		qh.Count = 2;
		if (FAILED(ctx.device->CreateQueryHeap(&qh, __uuidof(ID3D12QueryHeap), queries.ppv())))
			return false;
		readback = create(D3D12_HEAP_TYPE_READBACK, buffer(16, false), D3D12_RESOURCE_STATE_COPY_DEST);
		ctx.queue->GetTimestampFrequency(&frequency);
		return bool(readback);
	}

	ComPtr<ID3D12PipelineState> pipeline(const void *code, size_t size)
	{
		D3D12_COMPUTE_PIPELINE_STATE_DESC desc = {};
		desc.pRootSignature = root.get();
		desc.CS = { code, size };
		ComPtr<ID3D12PipelineState> pso;
		ctx.device->CreateComputePipelineState(&desc, __uuidof(ID3D12PipelineState), pso.ppv());
		return pso;
	}

	// Median GPU time in microseconds of `record` over several runs.
	double time_us(ID3D12PipelineState *pso, const UINT constants[4], const std::function<void(ID3D12GraphicsCommandList *)> &record)
	{
		std::vector<double> times;
		for (int run = 0; run < 7; run++)
		{
			ctx.begin();
			auto *cmd = ctx.list.get();
			ID3D12DescriptorHeap *heaps[] = { heap.get() };
			cmd->SetDescriptorHeaps(1, heaps);
			cmd->SetComputeRootSignature(root.get());
			cmd->SetPipelineState(pso);
			cmd->SetComputeRoot32BitConstants(0, 4, constants, 0);
			cmd->SetComputeRootUnorderedAccessView(1, result->GetGPUVirtualAddress());
			cmd->SetComputeRootShaderResourceView(2, source->GetGPUVirtualAddress());
			cmd->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
			cmd->EndQuery(queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
			record(cmd);
			cmd->EndQuery(queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
			cmd->ResolveQueryData(queries.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.get(), 0);
			ctx.wait(ctx.submit());
			UINT64 *ts = nullptr;
			D3D12_RANGE range = { 0, 16 };
			if (SUCCEEDED(readback->Map(0, &range, reinterpret_cast<void **>(&ts))))
			{
				times.push_back(double(ts[1] - ts[0]) * 1e6 / double(frequency));
				readback->Unmap(0, nullptr);
			}
		}
		std::sort(times.begin(), times.end());
		return times.empty() ? 0.0 : times[times.size() / 2];
	}
};
}

std::string run_tier_bench(Context &ctx)
{
	std::string report = "--- GPU partition micro-benchmarks ---\n";
	Bench bench(ctx);
	if (!bench.init())
		return report + "[FAIL] could not create the benchmark resources\n";

	auto latency = bench.pipeline(tier_bench_alu_latency, sizeof(tier_bench_alu_latency));
	auto throughput = bench.pipeline(tier_bench_alu_throughput, sizeof(tier_bench_alu_throughput));
	auto bandwidth = bench.pipeline(tier_bench_read_bandwidth, sizeof(tier_bench_read_bandwidth));
	auto store = bench.pipeline(tier_bench_typed_store, sizeof(tier_bench_typed_store));
	auto empty = bench.pipeline(tier_bench_empty_dispatch, sizeof(tier_bench_empty_dispatch));
	if (!latency || !throughput || !bandwidth || !store || !empty)
		return report + "[FAIL] could not create the benchmark pipelines\n";

	const float a = 0.999f, b = 0.001f;
	UINT c[4] = {};
	memcpy(&c[2], &a, 4);
	memcpy(&c[3], &b, 4);

	c[0] = 50000;
	double t = bench.time_us(latency.get(), c, [](ID3D12GraphicsCommandList *cmd) { cmd->Dispatch(1, 1, 1); });
	const double fma_ns = t * 1000.0 / (4.0 * c[0]);
	appendf(report, "dependent FMA chain (1 wave): %.3f ns per FMA = %.1f cycles at 1.825 GHz (%.0f us)\n", fma_ns,
	        fma_ns * 1.825, t);

	c[0] = 256;
	const UINT groups = 4096;
	t = bench.time_us(throughput.get(), c, [&](ID3D12GraphicsCommandList *cmd) { cmd->Dispatch(groups, 1, 1); });
	const double flops = double(groups) * 256.0 * 8.0 * 2.0 * c[0];
	appendf(report, "FMA throughput: %.2f TFLOPS (%.0f us; Series X peak 12.15)\n", flops / (t * 1e-6) / 1e12, t);

	c[0] = 16;
	c[1] = groups * 256;
	t = bench.time_us(bandwidth.get(), c, [&](ID3D12GraphicsCommandList *cmd) { cmd->Dispatch(groups, 1, 1); });
	const double bytes = double(groups) * 256.0 * 16.0 * c[0];
	appendf(report, "read bandwidth: %.0f GB/s (%.0f us; Series X peak 560)\n", bytes / (t * 1e-6) / 1e9, t);

	t = bench.time_us(store.get(), c, [](ID3D12GraphicsCommandList *cmd) { cmd->Dispatch(3840 / 64, 2160 / 8, 1); });
	appendf(report, "typed R16 stores (1 texel/lane): %.2f Gtexel/s (%.0f us for 3840x2160)\n",
	        3840.0 * 2160.0 / (t * 1e-6) / 1e9, t);

	t = bench.time_us(empty.get(), c, [](ID3D12GraphicsCommandList *cmd) {
		for (int i = 0; i < 100; i++)
			cmd->Dispatch(1, 1, 1);
	});
	appendf(report, "empty dispatch, no barriers: %.2f us each\n", t / 100.0);

	t = bench.time_us(empty.get(), c, [](ID3D12GraphicsCommandList *cmd) {
		D3D12_RESOURCE_BARRIER barrier = {};
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
		for (int i = 0; i < 100; i++)
		{
			cmd->Dispatch(1, 1, 1);
			cmd->ResourceBarrier(1, &barrier);
		}
	});
	appendf(report, "empty dispatch + UAV barrier: %.2f us each\n", t / 100.0);

	// Each of these moves 3840x2160 16-bit values (or dwords for the raw stores).
	struct Pattern
	{
		const char *name;
		const void *code;
		size_t size;
		UINT x, y, stride;
		double bytes_per_value;
	};
	const Pattern patterns[] = {
		{ "store R16, 1 texel/lane, coalesced rows", tier_bench_store_r16_rows, sizeof(tier_bench_store_r16_rows), 60, 270, 0, 2 },
		{ "store R16, iDWT output pattern", tier_bench_store_r16_idwt, sizeof(tier_bench_store_r16_idwt), 67, 120, 0, 2 },
		{ "store RG16, 2 values/lane", tier_bench_store_rg16_rows, sizeof(tier_bench_store_rg16_rows), 30, 270, 0, 2 },
		{ "store RGBA16, 4 values/lane", tier_bench_store_rgba16_rows, sizeof(tier_bench_store_rgba16_rows), 15, 270, 0, 2 },
		{ "store raw dword/lane", tier_bench_store_raw_dword, sizeof(tier_bench_store_raw_dword), 60, 270, 3840, 4 },
		{ "store raw dwordx4/lane", tier_bench_store_raw_dword4, sizeof(tier_bench_store_raw_dword4), 15, 270, 960, 4 },
		{ "load R16F via GatherRed (4 values/lane)", tier_bench_load_gather_r16, sizeof(tier_bench_load_gather_r16), 30, 135, 0, 2 },
		{ "load RGBA16F texel (4 values/lane)", tier_bench_load_rgba16, sizeof(tier_bench_load_rgba16), 15, 270, 0, 2 },
		{ "pyramid: R16F, 4x4 per lane", tier_bench_pyr_r16_lane4x4, sizeof(tier_bench_pyr_r16_lane4x4), 120, 67, 0, 2 },
		{ "pyramid: R16F, 8x8 per instruction", tier_bench_pyr_r16_tile8x8, sizeof(tier_bench_pyr_r16_tile8x8), 120, 67, 0, 2 },
		{ "pyramid: RGBA16 texels, 8x8 per instr", tier_bench_pyr_rgba16_tile, sizeof(tier_bench_pyr_rgba16_tile), 120, 67, 0, 2 },
		{ "pyramid: raw 2 KB tiles, 8 B/lane", tier_bench_pyr_raw_tiled_x2, sizeof(tier_bench_pyr_raw_tiled_x2), 120, 67, 120, 2 },
		{ "pyramid: raw 2 KB tiles, 16 B/lane", tier_bench_pyr_raw_tiled_x4, sizeof(tier_bench_pyr_raw_tiled_x4), 120, 67, 120, 2 },
		{ "pyramid: raw rows, 8 B/lane", tier_bench_pyr_raw_rows_x2, sizeof(tier_bench_pyr_raw_rows_x2), 120, 67, 0, 2 },
	};
	for (auto &p : patterns)
	{
		auto pso = bench.pipeline(p.code, p.size);
		if (!pso)
		{
			appendf(report, "%s: pipeline failed\n", p.name);
			continue;
		}
		c[1] = p.stride;
		t = bench.time_us(pso.get(), c, [&](ID3D12GraphicsCommandList *cmd) { cmd->Dispatch(p.x, p.y, 1); });
		appendf(report, "%-42s %7.0f us  %6.2f Gvalue/s  %5.0f GB/s\n", p.name, t, 3840.0 * 2160.0 / (t * 1e-6) / 1e9,
		        3840.0 * 2160.0 * p.bytes_per_value / (t * 1e-6) / 1e9);
	}
	return report;
}
}

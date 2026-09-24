// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "vector_test.hpp"

#include <dxgi1_4.h>

#include <algorithm>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace PyroWaveTest
{
namespace
{
constexpr uint32_t VectorMagic = 0x56545750; // "PWTV"
constexpr uint32_t VectorVersion = 1;

ComPtr<ID3D12Resource> create_resource(ID3D12Device *device, D3D12_HEAP_TYPE heap_type, const D3D12_RESOURCE_DESC &desc,
                                       D3D12_RESOURCE_STATES state)
{
	D3D12_HEAP_PROPERTIES heap = {};
	heap.Type = heap_type;
	ComPtr<ID3D12Resource> res;
	if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
	                                           __uuidof(ID3D12Resource), res.ppv())))
		return {};
	return res;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 size)
{
	D3D12_RESOURCE_DESC desc = {};
	desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	desc.Width = size;
	desc.Height = 1;
	desc.DepthOrArraySize = 1;
	desc.MipLevels = 1;
	desc.SampleDesc.Count = 1;
	desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	return desc;
}

bool write_u32(FILE *f, uint32_t v) { return fwrite(&v, 4, 1, f) == 1; }
bool read_u32(FILE *f, uint32_t &v) { return fread(&v, 4, 1, f) == 1; }
}

void appendf(std::string &out, const char *fmt, ...)
{
	char buf[1024];
	va_list va;
	va_start(va, fmt);
	vsnprintf(buf, sizeof(buf), fmt, va);
	va_end(va);
	out += buf;
}

void Planes::allocate(int w, int h, bool chroma_444)
{
	for (int i = 0; i < 3; i++)
	{
		width[i] = i == 0 || chroma_444 ? w : w / 2;
		height[i] = i == 0 || chroma_444 ? h : h / 2;
		data[i].resize(size_t(width[i]) * height[i]);
	}
}

Planes make_test_image(int width, int height, bool chroma_444)
{
	Planes p;
	p.allocate(width, height, chroma_444);

	// Gradients, sinusoids of several frequencies, hard edges, fine lines and a little
	// deterministic noise: something for every wavelet band.
	uint32_t rng = 1;
	for (int i = 0; i < 3; i++)
	{
		for (int y = 0; y < p.height[i]; y++)
		{
			for (int x = 0; x < p.width[i]; x++)
			{
				float fx = float(x) / float(p.width[i]);
				float fy = float(y) / float(p.height[i]);
				float v;
				if (i == 0)
				{
					v = 40.0f + 150.0f * fx;
					v += 30.0f * sinf(fx * 40.0f + fy * 7.0f);
					v += 12.0f * sinf(fy * 180.0f);
					if (((x / 24) + (y / 24)) & 1)
						v += fy < 0.5f ? 25.0f : -25.0f;
					if (y % 97 < 2 || x % 131 < 2)
						v = 235.0f;
				}
				else
				{
					v = 128.0f + 60.0f * sinf((i == 1 ? fx : fy) * 9.0f + float(i));
					if (fx > 0.6f && fy > 0.6f)
						v = i == 1 ? 60.0f : 200.0f;
				}
				rng = rng * 1664525u + 1013904223u;
				v += float((rng >> 24) & 7) - 3.5f;
				p.data[i][size_t(y) * p.width[i] + x] = uint8_t(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
			}
		}
	}
	return p;
}

bool write_test_vector(const std::wstring &path, const TestVector &vec)
{
	FILE *f = _wfopen(path.c_str(), L"wb");
	if (!f)
		return false;
	bool ok = write_u32(f, VectorMagic) && write_u32(f, VectorVersion) && write_u32(f, uint32_t(vec.width)) &&
	          write_u32(f, uint32_t(vec.height)) && write_u32(f, vec.chroma_444 ? 1u : 0u) &&
	          write_u32(f, uint32_t(vec.packets.size())) && write_u32(f, uint32_t(vec.bitstream.size())) &&
	          fwrite(vec.bitstream.data(), 1, vec.bitstream.size(), f) == vec.bitstream.size();
	for (auto &p : vec.packets)
		ok = ok && write_u32(f, uint32_t(p.offset)) && write_u32(f, uint32_t(p.size));
	for (auto &plane : vec.reference.data)
		ok = ok && fwrite(plane.data(), 1, plane.size(), f) == plane.size();
	return fclose(f) == 0 && ok;
}

bool read_test_vector(const std::wstring &path, TestVector &vec)
{
	FILE *f = _wfopen(path.c_str(), L"rb");
	if (!f)
		return false;
	uint32_t magic = 0, version = 0, w = 0, h = 0, c444 = 0, num_packets = 0, bitstream_size = 0;
	bool ok = read_u32(f, magic) && magic == VectorMagic && read_u32(f, version) && version == VectorVersion &&
	          read_u32(f, w) && read_u32(f, h) && read_u32(f, c444) && read_u32(f, num_packets) &&
	          read_u32(f, bitstream_size) && w && h && w <= 16384 && h <= 16384;
	if (ok)
	{
		vec.width = int(w);
		vec.height = int(h);
		vec.chroma_444 = c444 != 0;
		vec.bitstream.resize(bitstream_size);
		ok = fread(vec.bitstream.data(), 1, bitstream_size, f) == bitstream_size;
		vec.packets.resize(num_packets);
		for (auto &p : vec.packets)
		{
			uint32_t off = 0, size = 0;
			ok = ok && read_u32(f, off) && read_u32(f, size) && size_t(off) + size <= bitstream_size;
			p = { off, size };
		}
		vec.reference.allocate(vec.width, vec.height, vec.chroma_444);
		for (auto &plane : vec.reference.data)
			ok = ok && fread(plane.data(), 1, plane.size(), f) == plane.size();
	}
	fclose(f);
	return ok;
}

bool Context::init(bool debug, bool gpu_validation)
{
	if (debug)
	{
		ComPtr<ID3D12Debug1> dbg;
		if (SUCCEEDED(D3D12GetDebugInterface(__uuidof(ID3D12Debug1), dbg.ppv())))
		{
			dbg->EnableDebugLayer();
			// Validates descriptors and resource states as the shaders execute.
			dbg->SetEnableGPUBasedValidation(gpu_validation ? TRUE : FALSE);
		}
	}

	ComPtr<IDXGIFactory4> factory;
	ComPtr<IDXGIAdapter1> best;
	if (SUCCEEDED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), factory.ppv())))
	{
		SIZE_T best_vram = 0;
		for (UINT i = 0;; i++)
		{
			ComPtr<IDXGIAdapter1> adapter;
			if (factory->EnumAdapters1(i, reinterpret_cast<IDXGIAdapter1 **>(adapter.ppv())) == DXGI_ERROR_NOT_FOUND)
				break;
			DXGI_ADAPTER_DESC1 desc = {};
			adapter->GetDesc1(&desc);
			if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && (!best || desc.DedicatedVideoMemory > best_vram))
			{
				best_vram = desc.DedicatedVideoMemory;
				best = std::move(adapter);
				char name[128];
				snprintf(name, sizeof(name), "%ls", desc.Description);
				adapter_name = name;
			}
		}
	}

	if (FAILED(D3D12CreateDevice(best.get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), device.ppv())))
		return false;

	D3D12_COMMAND_QUEUE_DESC qd = {};
	qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	if (FAILED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), queue.ppv())) ||
	    FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
	                                          allocator.ppv())) ||
	    FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.get(), nullptr,
	                                     __uuidof(ID3D12GraphicsCommandList), list.ppv())) ||
	    FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.ppv())))
		return false;
	list->Close();
	event = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
	return event != nullptr;
}

Context::~Context()
{
	if (queue && fence)
		wait(fence_value);
	if (event)
		CloseHandle(event);
}

void Context::begin()
{
	allocator->Reset();
	list->Reset(allocator.get(), nullptr);
}

UINT64 Context::submit()
{
	list->Close();
	ID3D12CommandList *lists[] = { list.get() };
	queue->ExecuteCommandLists(1, lists);
	queue->Signal(fence.get(), ++fence_value);
	return fence_value;
}

void Context::wait(UINT64 value)
{
	if (fence->GetCompletedValue() < value)
	{
		fence->SetEventOnCompletion(value, event);
		WaitForSingleObject(event, INFINITE);
	}
}

int g_decode_pace_ms = 0;
DXGI_FORMAT g_output_format = DXGI_FORMAT_R8_UNORM;
std::vector<int> g_precisions = { 2, 1 };
std::wstring g_baseline_dir;
bool g_baseline_save = false;
bool g_baseline_quality = false;

bool decode(Context &ctx, pyrowave_d3d12_device device, const TestVector &vec, int iterations, Planes &out,
            DecodeStats &stats, std::string &error, const TestVector *previous)
{
	pyrowave_d3d12_decoder_create_info info = {};
	info.device = device;
	info.width = vec.width;
	info.height = vec.height;
	info.chroma = vec.chroma_444 ? PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444 : PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420;

	pyrowave_d3d12_decoder decoder = nullptr;
	pyrowave_d3d12_result res = pyrowave_d3d12_decoder_create(&info, &decoder);
	if (res != PYROWAVE_D3D12_SUCCESS)
	{
		error = std::string("pyrowave_d3d12_decoder_create: ") + pyrowave_d3d12_result_to_string(res);
		return false;
	}

	out.allocate(vec.width, vec.height, vec.chroma_444);
	ComPtr<ID3D12Resource> planes[3], readback[3];
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[3] = {};
	for (int i = 0; i < 3; i++)
	{
		D3D12_RESOURCE_DESC desc = {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Width = UINT64(out.width[i]);
		desc.Height = UINT(out.height[i]);
		desc.DepthOrArraySize = 1;
		desc.MipLevels = 1;
		desc.Format = g_output_format;
		desc.SampleDesc.Count = 1;
		desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		planes[i] = create_resource(ctx.device.get(), D3D12_HEAP_TYPE_DEFAULT, desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

		UINT64 total = 0;
		ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprints[i], nullptr, nullptr, &total);
		readback[i] = create_resource(ctx.device.get(), D3D12_HEAP_TYPE_READBACK, buffer_desc(total),
		                              D3D12_RESOURCE_STATE_COPY_DEST);
		if (!planes[i] || !readback[i])
		{
			error = "failed to allocate output planes";
			pyrowave_d3d12_decoder_destroy(decoder);
			return false;
		}
	}

	ComPtr<ID3D12QueryHeap> timestamps;
	D3D12_QUERY_HEAP_DESC qh = {};
	qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	qh.Count = 2;
	ctx.device->CreateQueryHeap(&qh, __uuidof(ID3D12QueryHeap), timestamps.ppv());
	auto ts_readback = create_resource(ctx.device.get(), D3D12_HEAP_TYPE_READBACK, buffer_desc(16),
	                                   D3D12_RESOURCE_STATE_COPY_DEST);
	UINT64 ts_freq = 1;
	ctx.queue->GetTimestampFrequency(&ts_freq);

	std::vector<double> times;
	for (int iter = previous ? -1 : 0; iter < iterations; iter++)
	{
		// Iteration -1 decodes `previous`.
		const TestVector &frame = iter < 0 ? *previous : vec;
		// Re-push every iteration: a decode marks the frame consumed.
		pyrowave_d3d12_decoder_clear(decoder);
		for (auto &p : frame.packets)
		{
			res = pyrowave_d3d12_decoder_push_packet(decoder, frame.bitstream.data() + p.offset, p.size);
			if (res != PYROWAVE_D3D12_SUCCESS)
			{
				error = std::string("pyrowave_d3d12_decoder_push_packet: ") + pyrowave_d3d12_result_to_string(res);
				pyrowave_d3d12_decoder_destroy(decoder);
				return false;
			}
		}
		if (!pyrowave_d3d12_decoder_decode_is_ready(decoder, false))
		{
			error = "frame not ready after pushing every packet";
			pyrowave_d3d12_decoder_destroy(decoder);
			return false;
		}

		ctx.begin();
		ctx.list->EndQuery(timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
		pyrowave_d3d12_gpu_buffers buffers = { { planes[0].get(), planes[1].get(), planes[2].get() } };
		res = pyrowave_d3d12_decoder_decode_gpu_buffer(decoder, ctx.list.get(), &buffers, ctx.fence.get(),
		                                               ctx.fence_value + 1);
		if (res != PYROWAVE_D3D12_SUCCESS)
		{
			error = std::string("pyrowave_d3d12_decoder_decode_gpu_buffer: ") + pyrowave_d3d12_result_to_string(res);
			ctx.list->Close();
			pyrowave_d3d12_decoder_destroy(decoder);
			return false;
		}
		ctx.list->EndQuery(timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
		ctx.list->ResolveQueryData(timestamps.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts_readback.get(), 0);

		if (iter == iterations - 1)
		{
			D3D12_RESOURCE_BARRIER barriers[3] = {};
			for (int i = 0; i < 3; i++)
			{
				barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
				barriers[i].Transition.pResource = planes[i].get();
				barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
				barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
			}
			ctx.list->ResourceBarrier(3, barriers);
			for (int i = 0; i < 3; i++)
			{
				D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
				dst.pResource = readback[i].get();
				dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
				dst.PlacedFootprint = footprints[i];
				src.pResource = planes[i].get();
				src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
				ctx.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
			}
		}

		ctx.wait(ctx.submit());
		if (g_decode_pace_ms > 0)
			Sleep(DWORD(g_decode_pace_ms));
		HRESULT removed = ctx.device->GetDeviceRemovedReason();
		if (FAILED(removed))
		{
			char buf[64];
			snprintf(buf, sizeof(buf), "device removed (hr 0x%08lx)", static_cast<unsigned long>(removed));
			error = buf;
			pyrowave_d3d12_decoder_destroy(decoder);
			return false;
		}

		UINT64 *ts = nullptr;
		D3D12_RANGE range = { 0, 16 };
		if (iter >= 0 && SUCCEEDED(ts_readback->Map(0, &range, reinterpret_cast<void **>(&ts))))
		{
			times.push_back(double(ts[1] - ts[0]) * 1000.0 / double(ts_freq));
			ts_readback->Unmap(0, nullptr);
		}
	}

	std::sort(times.begin(), times.end());
	stats.best_ms = times.empty() ? 0.0 : times.front();
	stats.median_ms = times.empty() ? 0.0 : times[times.size() / 2];

	const bool wide = g_output_format == DXGI_FORMAT_R16_UNORM;
	for (int i = 0; i < 3; i++)
	{
		uint8_t *mapped = nullptr;
		readback[i]->Map(0, nullptr, reinterpret_cast<void **>(&mapped));
		if (wide)
			out.data16[i].resize(out.data[i].size());
		for (int y = 0; y < out.height[i]; y++)
		{
			const uint8_t *row = mapped + footprints[i].Offset + size_t(y) * footprints[i].Footprint.RowPitch;
			const size_t base = size_t(y) * out.width[i];
			if (!wide)
			{
				memcpy(out.data[i].data() + base, row, size_t(out.width[i]));
				continue;
			}
			memcpy(out.data16[i].data() + base, row, size_t(out.width[i]) * 2);
			// UNORM16 -> UNORM8 with rounding, as a shader writing R8 would.
			for (int x = 0; x < out.width[i]; x++)
				out.data[i][base + x] = uint8_t((uint32_t(out.data16[i][base + x]) * 255u + 32767u) / 65535u);
		}
		readback[i]->Unmap(0, nullptr);
	}

	pyrowave_d3d12_decoder_destroy(decoder);
	return true;
}

PlaneDiff compare(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
	PlaneDiff d;
	d.total = a.size();
	for (size_t i = 0; i < a.size() && i < b.size(); i++)
	{
		int diff = abs(int(a[i]) - int(b[i]));
		if (diff)
			d.mismatches++;
		if (diff > d.max_diff)
			d.max_diff = diff;
	}
	return d;
}

double psnr(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
	double mse = 0.0;
	for (size_t i = 0; i < a.size() && i < b.size(); i++)
	{
		double d = double(a[i]) - double(b[i]);
		mse += d * d;
	}
	mse /= double(a.size() ? a.size() : 1);
	return mse == 0.0 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

double ssim(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, int width, int height)
{
	constexpr int R = 5;
	float w[2 * R + 1];
	float sum = 0.0f;
	for (int i = -R; i <= R; i++)
		sum += (w[i + R] = expf(-float(i * i) / (2.0f * 1.5f * 1.5f)));
	for (auto &v : w)
		v /= sum;
	if (width <= 2 * R || height <= 2 * R)
		return 1.0;

	// Separable filtering of x, y, x^2, y^2 and xy; the horizontal pass keeps only the
	// columns the window fits around.
	const int ow = width - 2 * R;
	std::vector<float> h[5];
	for (auto &plane : h)
		plane.resize(size_t(ow) * height);
	for (int y = 0; y < height; y++)
	{
		const uint8_t *ra = a.data() + size_t(y) * width;
		const uint8_t *rb = b.data() + size_t(y) * width;
		for (int x = 0; x < ow; x++)
		{
			float s[5] = {};
			for (int k = 0; k <= 2 * R; k++)
			{
				const float fa = ra[x + k], fb = rb[x + k];
				s[0] += w[k] * fa;
				s[1] += w[k] * fb;
				s[2] += w[k] * fa * fa;
				s[3] += w[k] * fb * fb;
				s[4] += w[k] * fa * fb;
			}
			for (int i = 0; i < 5; i++)
				h[i][size_t(y) * ow + x] = s[i];
		}
	}

	const double c1 = (0.01 * 255.0) * (0.01 * 255.0), c2 = (0.03 * 255.0) * (0.03 * 255.0);
	double total = 0.0;
	for (int y = 0; y < height - 2 * R; y++)
	{
		for (int x = 0; x < ow; x++)
		{
			double s[5] = {};
			for (int k = 0; k <= 2 * R; k++)
				for (int i = 0; i < 5; i++)
					s[i] += w[k] * h[i][size_t(y + k) * ow + x];
			const double va = s[2] - s[0] * s[0], vb = s[3] - s[1] * s[1], cov = s[4] - s[0] * s[1];
			total += ((2.0 * s[0] * s[1] + c1) * (2.0 * cov + c2)) /
			         ((s[0] * s[0] + s[1] * s[1] + c1) * (va + vb + c2));
		}
	}
	return total / (double(ow) * double(height - 2 * R));
}

bool read_raw_planes(const char *path, Planes &p)
{
	FILE *f = fopen(path, "rb");
	if (!f)
		return false;
	bool ok = true;
	for (auto &plane : p.data)
		ok = ok && fread(plane.data(), 1, plane.size(), f) == plane.size();
	fclose(f);
	return ok;
}

// Collects the decoder's per-stage profile lines (PYROWAVE_D3D12_PROFILE=1) and which
// kernels the device picked.
struct DecoderLog
{
	std::vector<std::string> profile_lines;
	std::string kernels;
};

static void keep_profile(void *userdata, const char *msg)
{
	auto *log = static_cast<DecoderLog *>(userdata);
	if (strstr(msg, "decode profile"))
		log->profile_lines.push_back(msg);
	else if (strstr(msg, "Decoder kernels"))
		log->kernels = msg;

}

// "decode profile total T us: a 1.0 b 2.0 ..." lines -> one line of per-stage medians.
static std::string median_profile(const std::vector<std::string> &lines)
{
	std::vector<std::string> names;
	std::vector<std::vector<double>> values;
	for (auto &line : lines)
	{
		size_t colon = line.find("us:");
		if (colon == std::string::npos)
			continue;
		std::vector<std::string> tokens;
		size_t pos = colon + 3;
		while (pos < line.size())
		{
			while (pos < line.size() && line[pos] == ' ')
				pos++;
			size_t end = line.find(' ', pos);
			if (end == std::string::npos)
				end = line.size();
			if (end > pos)
				tokens.push_back(line.substr(pos, end - pos));
			pos = end;
		}
		std::vector<std::string> line_names;
		std::vector<double> line_values;
		line_names.push_back("total");
		line_values.push_back(atof(line.c_str() + line.find("total") + 5));
		for (size_t i = 0; i + 1 < tokens.size(); i += 2)
		{
			line_names.push_back(tokens[i]);
			line_values.push_back(atof(tokens[i + 1].c_str()));
		}
		if (names.empty())
		{
			names = line_names;
			values.resize(names.size());
		}
		if (line_names != names)
			continue;
		for (size_t i = 0; i < names.size(); i++)
			values[i].push_back(line_values[i]);
	}
	if (names.empty())
		return {};
	std::string out;
	appendf(out, "stage medians (us, %zu frames):", values[0].size());
	for (size_t i = 0; i < names.size(); i++)
	{
		std::sort(values[i].begin(), values[i].end());
		appendf(out, " %s %.1f", names[i].c_str(), values[i][values[i].size() / 2]);
	}
	return out;
}

// Saves or compares the raw output planes against g_baseline_dir.
static std::string check_baseline(const TestVector &vec, int precision, const Planes &out)
{
	const bool wide = !out.data16[0].empty();
	std::wstring path = g_baseline_dir + L"\\";
	for (char c : vec.name)
		path += wchar_t(c);
	path += L"_p" + std::to_wstring(g_baseline_quality ? 2 : precision) + (wide ? L"_16" : L"_8") + L".raw";

	std::vector<uint8_t> bytes;
	for (int i = 0; i < 3; i++)
	{
		const uint8_t *p = wide ? reinterpret_cast<const uint8_t *>(out.data16[i].data()) : out.data[i].data();
		bytes.insert(bytes.end(), p, p + out.data[i].size() * (wide ? 2 : 1));
	}

	if (g_baseline_save)
	{
		FILE *f = _wfopen(path.c_str(), L"wb");
		bool ok = f && fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
		if (f)
			fclose(f);
		return ok ? "baseline saved" : "baseline SAVE FAILED";
	}

	std::vector<uint8_t> ref(bytes.size());
	FILE *f = _wfopen(path.c_str(), L"rb");
	if (!f)
		return "baseline MISSING";
	bool ok = fread(ref.data(), 1, ref.size(), f) == ref.size();
	fclose(f);
	if (!ok)
		return "baseline MISSING (short file)";

	size_t differ = 0;
	int max_diff = 0;
	double sum_sq = 0.0;
	const size_t elem = wide ? 2 : 1;
	for (size_t i = 0; i < bytes.size(); i += elem)
	{
		int a = wide ? int(bytes[i] | (bytes[i + 1] << 8)) : int(bytes[i]);
		int b = wide ? int(ref[i] | (ref[i + 1] << 8)) : int(ref[i]);
		if (a != b)
		{
			differ++;
			max_diff = abs(a - b) > max_diff ? abs(a - b) : max_diff;
			sum_sq += double(a - b) * double(a - b);
		}
	}
	std::string r;
	if (g_baseline_quality)
	{
		const double peak = wide ? 65535.0 : 255.0;
		const double mse = sum_sq / double(bytes.size() / elem);
		appendf(r, "vs FP32 decode (quality): max diff %d, rms %.2f, PSNR %.2f dB (%s units)", max_diff, sqrt(mse),
		        mse > 0.0 ? 10.0 * log10(peak * peak / mse) : 99.0, wide ? "16-bit" : "8-bit");
		return r;
	}
	if (differ == 0)
		r = "baseline IDENTICAL";
	else
		appendf(r, "baseline DIFFERS: %zu of %zu values, max diff %d", differ, bytes.size() / elem, max_diff);
	return r;
}

std::string run_suite(Context &ctx, const std::vector<TestVector> &vectors, int iterations, bool &pass)
{
	DecoderLog log;
	std::string report;
	pass = true;
	appendf(report, "Adapter: %s, output %s\n", ctx.adapter_name.c_str(),
	        g_output_format == DXGI_FORMAT_R16_UNORM ? "R16_UNORM" : "R8_UNORM");

	for (int precision : g_precisions)
	{
		pyrowave_d3d12_device_create_info info = {};
		info.d3d12_device = ctx.device.get();
		info.wavelet_precision = precision;
		info.message_callback = keep_profile;
		info.message_userdata = &log;
		pyrowave_d3d12_device device = nullptr;
		pyrowave_d3d12_result res = pyrowave_d3d12_device_create(&info, &device);
		if (res != PYROWAVE_D3D12_SUCCESS)
		{
			appendf(report, "[FAIL] pyrowave_d3d12_device_create(precision %d): %s\n", precision,
			        pyrowave_d3d12_result_to_string(res));
			pass = false;
			continue;
		}

		const int tolerance = precision == 2 ? 1 : 2;
		appendf(report, "\n--- precision %d (%s), tolerance %d ---\n%s\n", precision,
		        precision == 2 ? "FP32" : "FP16 storage", tolerance, log.kernels.c_str());
		for (auto &vec : vectors)
		{
			Planes out;
			DecodeStats stats;
			std::string error;
			// Decode another frame of the same format first, the densest one, so stale
			// pyramid contents are another frame's.
			const TestVector *previous = nullptr;
			for (auto &other : vectors)
				if (&other != &vec && other.width == vec.width && other.height == vec.height &&
				    other.chroma_444 == vec.chroma_444 &&
				    (!previous || other.bitstream.size() > previous->bitstream.size()))
					previous = &other;
			log.profile_lines.clear();
			if (!decode(ctx, device, vec, iterations, out, stats, error, previous))
			{
				appendf(report, "[FAIL] %s: %s\n", vec.name.c_str(), error.c_str());
				pass = false;
				continue;
			}

			int worst = 0;
			size_t mismatches = 0, total = 0;
			double worst_psnr = 99.0;
			for (int i = 0; i < 3; i++)
			{
				PlaneDiff d = compare(vec.reference.data[i], out.data[i]);
				worst = d.max_diff > worst ? d.max_diff : worst;
				mismatches += d.mismatches;
				total += d.total;
				double p = psnr(vec.reference.data[i], out.data[i]);
				worst_psnr = p < worst_psnr ? p : worst_psnr;
			}
			bool ok = worst <= tolerance;
			std::string baseline;
			if (!g_baseline_dir.empty())
			{
				baseline = check_baseline(vec, precision, out);
				if (!g_baseline_quality && baseline.find("IDENTICAL") == std::string::npos &&
				    baseline.find("saved") == std::string::npos)
					ok = false;
			}
			pass = pass && ok;
			appendf(report, "[%s] %-16s decode %.3f ms (median %.3f)  max diff %d, %.3f%% px differ, PSNR vs ref %.1f dB\n",
			        ok ? "PASS" : "FAIL", vec.name.c_str(), stats.best_ms, stats.median_ms, worst,
			        100.0 * double(mismatches) / double(total ? total : 1), worst_psnr);
			if (previous)
				appendf(report, "    (decoded after %s)\n", previous->name.c_str());
			if (!baseline.empty())
				appendf(report, "    %s\n", baseline.c_str());
			std::string profile = median_profile(log.profile_lines);
			if (!profile.empty())
				appendf(report, "    %s\n", profile.c_str());
		}
		pyrowave_d3d12_device_destroy(device);
	}

	appendf(report, "\nOverall: %s\n", pass ? "PASS" : "FAIL");
	return report;
}
}

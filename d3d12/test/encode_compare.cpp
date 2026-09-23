// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Validates the D3D12 encoder against the Vulkan one.
//
// For each synthetic frame and rate target:
//   A  Vulkan encode   -> Vulkan decode   (reference)
//   B  D3D12 encode    -> Vulkan decode   (bitstream compatibility + quality)
//   C  D3D12 encode    -> D3D12 decode    (must match B within rounding)
//   D  D3D12 GPU input -> Vulkan decode   (must equal B exactly: same data, same GPU)
//   E  D3D12 NV12 CPU input (420 only)    (must equal B exactly)
// Bitstreams are never compared byte for byte (the packer leaves uninitialized bits in
// the last sign byte of a block, as documented in the Vulkan and Metal encoders);
// decoded output is.
//
// Usage: pyrowave-d3d12-encode-compare [--precision N] [--debug | --gbv] [--iterations N]

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "pyrowave.h"

#include "vector_test.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

using namespace PyroWaveTest;

namespace
{
struct TestCase
{
	int width;
	int height;
	bool chroma_444;
};

const TestCase test_cases[] = {
	{ 1920, 1080, false },
	{ 1920, 1080, true },
	{ 1366, 768, true },
	{ 3840, 2160, true },
	{ 3840, 2160, false },
	{ 640, 360, false },
};

// Bits per luma pixel.
const double rates[] = { 3.0, 1.0 };

struct Encoded
{
	std::vector<uint8_t> bitstream;
	std::vector<PacketSpan> packets;
	size_t bytes = 0;
};

pyrowave_cpu_buffer vk_cpu_buffer(Planes &p, bool chroma_444)
{
	pyrowave_cpu_buffer buf = {};
	for (int i = 0; i < 3; i++)
	{
		buf.data[i] = p.data[i].data();
		buf.row_stride_in_bytes[i] = size_t(p.width[i]);
		buf.plane_size_in_bytes[i] = p.data[i].size();
	}
	buf.width = p.width[0];
	buf.height = p.height[0];
	buf.format = chroma_444 ? PYROWAVE_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
	return buf;
}

void finish_packets(Encoded &enc, const size_t *offsets, const size_t *sizes, size_t count)
{
	size_t end = 0;
	enc.packets.clear();
	enc.bytes = 0;
	for (size_t i = 0; i < count; i++)
	{
		enc.packets.push_back({ offsets[i], sizes[i] });
		enc.bytes += sizes[i];
		end = offsets[i] + sizes[i] > end ? offsets[i] + sizes[i] : end;
	}
	enc.bitstream.resize(end);
}

constexpr size_t PacketBoundary = 16 * 1024;

bool vk_encode(pyrowave_device vk, const TestCase &tc, Planes &source, size_t target, Encoded &out)
{
	pyrowave_encoder_create_info info = {};
	info.device = vk;
	info.width = tc.width;
	info.height = tc.height;
	info.chroma = tc.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_encoder encoder = nullptr;
	if (pyrowave_encoder_create(&info, &encoder) != PYROWAVE_SUCCESS)
		return false;
	pyrowave_rate_control rc = { target };
	pyrowave_cpu_buffer buf = vk_cpu_buffer(source, tc.chroma_444);
	size_t num = 0, written = 0;
	bool ok = pyrowave_encoder_encode_cpu_synchronous(encoder, &buf, &rc) == PYROWAVE_SUCCESS &&
	          pyrowave_encoder_compute_num_packets(encoder, PacketBoundary, &num) == PYROWAVE_SUCCESS;
	std::vector<pyrowave_packet> packets(num);
	out.bitstream.resize(target * 2 + 64 * 1024);
	ok = ok && pyrowave_encoder_packetize(encoder, packets.data(), PacketBoundary, &written, out.bitstream.data(),
	                                      out.bitstream.size()) == PYROWAVE_SUCCESS;
	pyrowave_encoder_destroy(encoder);
	std::vector<size_t> offs, sizes;
	for (size_t i = 0; i < written; i++)
	{
		offs.push_back(packets[i].offset);
		sizes.push_back(packets[i].size);
	}
	finish_packets(out, offs.data(), sizes.data(), written);
	return ok;
}

bool vk_decode(pyrowave_device vk, const TestCase &tc, const Encoded &enc, Planes &out)
{
	pyrowave_decoder_create_info info = {};
	info.device = vk;
	info.width = tc.width;
	info.height = tc.height;
	info.chroma = tc.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;
	pyrowave_decoder decoder = nullptr;
	if (pyrowave_decoder_create(&info, &decoder) != PYROWAVE_SUCCESS)
		return false;
	bool ok = true;
	for (auto &p : enc.packets)
		ok = ok && pyrowave_decoder_push_packet(decoder, enc.bitstream.data() + p.offset, p.size) == PYROWAVE_SUCCESS;
	ok = ok && pyrowave_decoder_decode_is_ready(decoder, false);
	out.allocate(tc.width, tc.height, tc.chroma_444);
	pyrowave_cpu_buffer buf = vk_cpu_buffer(out, tc.chroma_444);
	ok = ok && pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &buf) == PYROWAVE_SUCCESS;
	pyrowave_decoder_destroy(decoder);
	return ok;
}

bool d3d12_collect(pyrowave_d3d12_encoder encoder, size_t target, Encoded &out)
{
	size_t num = 0, written = 0;
	if (pyrowave_d3d12_encoder_compute_num_packets(encoder, PacketBoundary, &num) != PYROWAVE_D3D12_SUCCESS)
		return false;
	std::vector<pyrowave_d3d12_packet> packets(num);
	out.bitstream.resize(target * 2 + 64 * 1024);
	if (pyrowave_d3d12_encoder_packetize(encoder, packets.data(), PacketBoundary, &written, out.bitstream.data(),
	                                     out.bitstream.size()) != PYROWAVE_D3D12_SUCCESS)
		return false;
	std::vector<size_t> offs, sizes;
	for (size_t i = 0; i < written; i++)
	{
		offs.push_back(packets[i].offset);
		sizes.push_back(packets[i].size);
	}
	finish_packets(out, offs.data(), sizes.data(), written);
	return true;
}

// R8 textures holding the source planes, in NON_PIXEL_SHADER_RESOURCE.
bool upload_planes(Context &ctx, const Planes &p, ComPtr<ID3D12Resource> textures[3])
{
	ComPtr<ID3D12Resource> uploads[3];
	ctx.begin();
	for (int i = 0; i < 3; i++)
	{
		D3D12_RESOURCE_DESC desc = {};
		desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		desc.Width = UINT64(p.width[i]);
		desc.Height = UINT(p.height[i]);
		desc.DepthOrArraySize = 1;
		desc.MipLevels = 1;
		desc.Format = DXGI_FORMAT_R8_UNORM;
		desc.SampleDesc.Count = 1;
		D3D12_HEAP_PROPERTIES heap = {};
		heap.Type = D3D12_HEAP_TYPE_DEFAULT;
		if (FAILED(ctx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
		                                               nullptr, __uuidof(ID3D12Resource), textures[i].ppv())))
			return false;

		D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
		UINT64 total = 0;
		ctx.device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
		D3D12_RESOURCE_DESC bd = {};
		bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		bd.Width = total;
		bd.Height = 1;
		bd.DepthOrArraySize = 1;
		bd.MipLevels = 1;
		bd.SampleDesc.Count = 1;
		bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
		heap.Type = D3D12_HEAP_TYPE_UPLOAD;
		if (FAILED(ctx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ,
		                                               nullptr, __uuidof(ID3D12Resource), uploads[i].ppv())))
			return false;
		uint8_t *mapped = nullptr;
		uploads[i]->Map(0, nullptr, reinterpret_cast<void **>(&mapped));
		for (int y = 0; y < p.height[i]; y++)
			memcpy(mapped + fp.Offset + size_t(y) * fp.Footprint.RowPitch, p.data[i].data() + size_t(y) * p.width[i],
			       size_t(p.width[i]));
		uploads[i]->Unmap(0, nullptr);

		D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
		dst.pResource = textures[i].get();
		dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
		src.pResource = uploads[i].get();
		src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
		src.PlacedFootprint = fp;
		ctx.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

		D3D12_RESOURCE_BARRIER b = {};
		b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		b.Transition.pResource = textures[i].get();
		b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		ctx.list->ResourceBarrier(1, &b);
	}
	ctx.wait(ctx.submit());
	return true;
}

double avg_psnr(const Planes &a, const Planes &b)
{
	// Luma-weighted like most codec reports: (6Y + Cb + Cr) / 8.
	return (6.0 * psnr(a.data[0], b.data[0]) + psnr(a.data[1], b.data[1]) + psnr(a.data[2], b.data[2])) / 8.0;
}

int max_diff(const Planes &a, const Planes &b)
{
	int worst = 0;
	for (int i = 0; i < 3; i++)
	{
		PlaneDiff d = compare(a.data[i], b.data[i]);
		worst = d.max_diff > worst ? d.max_diff : worst;
	}
	return worst;
}

size_t report_debug_layer(Context &ctx)
{
	size_t issues = 0;
	ComPtr<ID3D12InfoQueue> info_queue;
	if (FAILED(ctx.device->QueryInterface(__uuidof(ID3D12InfoQueue), info_queue.ppv())))
		return 0;
	const UINT64 count = info_queue->GetNumStoredMessages();
	for (UINT64 i = 0; i < count; i++)
	{
		SIZE_T len = 0;
		info_queue->GetMessage(i, nullptr, &len);
		std::vector<uint8_t> storage(len);
		auto *msg = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
		info_queue->GetMessage(i, msg, &len);
		if (msg->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
		{
			issues++;
			fprintf(stderr, "D3D12 debug layer: %s\n", msg->pDescription);
		}
	}
	printf("\nD3D12 debug layer: %zu warnings/errors in %llu stored messages.\n", issues,
	       static_cast<unsigned long long>(count));
	return issues;
}
}

int main(int argc, char **argv)
{
	int precision = 2;
	int iterations = 10;
	bool debug = false, gpu_validation = false;
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--precision") && i + 1 < argc)
			precision = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iterations") && i + 1 < argc)
			iterations = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--debug"))
			debug = true;
		else if (!strcmp(argv[i], "--gbv"))
			debug = gpu_validation = true;
	}

	char env[32];
	snprintf(env, sizeof(env), "PYROWAVE_PRECISION=%d", precision);
	_putenv(env);

	pyrowave_device vk = nullptr;
	if (pyrowave_create_default_device(&vk) != PYROWAVE_SUCCESS)
	{
		fprintf(stderr, "Failed to create the Vulkan PyroWave device.\n");
		return 1;
	}

	Context ctx;
	if (!ctx.init(debug, gpu_validation))
	{
		fprintf(stderr, "Failed to create the D3D12 device.\n");
		return 1;
	}

	pyrowave_d3d12_device_create_info info = {};
	info.d3d12_device = ctx.device.get();
	info.wavelet_precision = precision == 2 ? 2 : 1;
	pyrowave_d3d12_device d3d12 = nullptr;
	if (pyrowave_d3d12_device_create(&info, &d3d12) != PYROWAVE_D3D12_SUCCESS)
	{
		fprintf(stderr, "pyrowave_d3d12_device_create failed.\n");
		return 1;
	}

	printf("D3D12 adapter: %s, PYROWAVE_PRECISION=%d%s\n\n", ctx.adapter_name.c_str(), precision,
	       gpu_validation ? ", debug layer + GPU-based validation on" : (debug ? ", debug layer on" : ""));
	printf("%-14s %4s  %9s %9s %9s  %7s %7s %6s  %4s %4s %4s  %8s\n", "case", "bpp", "target", "vk bytes", "d12 bytes",
	       "vk dB", "d12 dB", "delta", "C", "D", "E", "enc ms");

	bool ok = true;
	for (auto &tc : test_cases)
	{
		Planes source = make_test_image(tc.width, tc.height, tc.chroma_444);
		ComPtr<ID3D12Resource> textures[3];
		if (!upload_planes(ctx, source, textures))
		{
			fprintf(stderr, "Failed to upload input textures.\n");
			return 1;
		}

		pyrowave_d3d12_encoder_create_info enc_info = {};
		enc_info.device = d3d12;
		enc_info.width = tc.width;
		enc_info.height = tc.height;
		enc_info.chroma = tc.chroma_444 ? PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444 : PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420;
		pyrowave_d3d12_encoder encoder = nullptr;
		pyrowave_d3d12_result res = pyrowave_d3d12_encoder_create(&enc_info, &encoder);
		if (res != PYROWAVE_D3D12_SUCCESS)
		{
			fprintf(stderr, "pyrowave_d3d12_encoder_create: %s\n", pyrowave_d3d12_result_to_string(res));
			return 1;
		}

		for (double bpp : rates)
		{
			const size_t target = size_t(double(tc.width) * tc.height * bpp / 8.0);
			char name[32];
			snprintf(name, sizeof(name), "%dx%d_%s", tc.width, tc.height, tc.chroma_444 ? "444" : "420");

			// A: Vulkan reference.
			Encoded a;
			Planes dec_a;
			if (!vk_encode(vk, tc, source, target, a) || !vk_decode(vk, tc, a, dec_a))
			{
				printf("%-14s %4.1f  Vulkan reference failed\n", name, bpp);
				ok = false;
				continue;
			}

			// B: D3D12 encode (CPU input), Vulkan decode.
			pyrowave_d3d12_rate_control rc = { target };
			pyrowave_d3d12_cpu_buffer cpu = {};
			for (int i = 0; i < 3; i++)
			{
				cpu.data[i] = source.data[i].data();
				cpu.row_stride_in_bytes[i] = size_t(source.width[i]);
				cpu.plane_size_in_bytes[i] = source.data[i].size();
			}
			cpu.width = tc.width;
			cpu.height = tc.height;
			cpu.format = tc.chroma_444 ? PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV444P : PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV420P;
			Encoded b;
			Planes dec_b;
			res = pyrowave_d3d12_encoder_encode_cpu(encoder, &cpu, &rc);
			bool b_ok = res == PYROWAVE_D3D12_SUCCESS && d3d12_collect(encoder, target, b) && vk_decode(vk, tc, b, dec_b);
			if (!b_ok)
			{
				printf("%-14s %4.1f  D3D12 encode -> Vulkan decode FAILED (%s)\n", name, bpp,
				       pyrowave_d3d12_result_to_string(res));
				ok = false;
				continue;
			}

			// C: the same bitstream through the D3D12 decoder.
			TestVector vec;
			vec.width = tc.width;
			vec.height = tc.height;
			vec.chroma_444 = tc.chroma_444;
			vec.bitstream = b.bitstream;
			vec.packets = b.packets;
			Planes dec_c;
			DecodeStats dstats;
			std::string error;
			int c_diff = decode(ctx, d3d12, vec, 1, dec_c, dstats, error) ? max_diff(dec_b, dec_c) : 999;

			// D: GPU texture input, timed over several encodes.
			pyrowave_d3d12_encoder_gpu_input gin = {};
			for (int i = 0; i < 3; i++)
				gin.planes[i] = textures[i].get();
			double best_ms = 1e9;
			Encoded d;
			Planes dec_d;
			bool d_ok = true;
			for (int it = 0; it < iterations && d_ok; it++)
			{
				d_ok = pyrowave_d3d12_encoder_encode_gpu(encoder, &gin, &rc) == PYROWAVE_D3D12_SUCCESS;
				double ms = 0.0;
				if (d_ok && pyrowave_d3d12_encoder_get_last_gpu_time(encoder, &ms) == PYROWAVE_D3D12_SUCCESS &&
				    ms < best_ms)
					best_ms = ms;
			}
			d_ok = d_ok && d3d12_collect(encoder, target, d) && vk_decode(vk, tc, d, dec_d);
			int d_diff = d_ok ? max_diff(dec_b, dec_d) : 999;

			// E: NV12 CPU input (420 only).
			int e_diff = -1;
			if (!tc.chroma_444)
			{
				std::vector<uint8_t> uv(source.data[1].size() * 2);
				for (size_t i = 0; i < source.data[1].size(); i++)
				{
					uv[2 * i + 0] = source.data[1][i];
					uv[2 * i + 1] = source.data[2][i];
				}
				pyrowave_d3d12_cpu_buffer nv12 = cpu;
				nv12.format = PYROWAVE_D3D12_CPU_BUFFER_FORMAT_NV12;
				nv12.data[1] = uv.data();
				nv12.row_stride_in_bytes[1] = size_t(source.width[1]) * 2;
				nv12.plane_size_in_bytes[1] = uv.size();
				nv12.data[2] = nullptr;
				Encoded e;
				Planes dec_e;
				bool e_ok = pyrowave_d3d12_encoder_encode_cpu(encoder, &nv12, &rc) == PYROWAVE_D3D12_SUCCESS &&
				            d3d12_collect(encoder, target, e) && vk_decode(vk, tc, e, dec_e);
				e_diff = e_ok ? max_diff(dec_b, dec_e) : 999;
			}

			const double psnr_a = avg_psnr(source, dec_a);
			const double psnr_b = avg_psnr(source, dec_b);
			const double delta = psnr_b - psnr_a;
			const int c_tol = precision == 2 ? 1 : 2;
			const bool row_ok = b.bytes <= target && delta > -0.25 && c_diff <= c_tol && d_diff == 0 && e_diff <= 0;
			ok = ok && row_ok;

			char e_str[8];
			snprintf(e_str, sizeof(e_str), "%d", e_diff);
			printf("%-14s %4.1f  %9zu %9zu %9zu  %7.2f %7.2f %+6.2f  %4d %4d %4s  %8.3f  %s\n", name, bpp, target, a.bytes,
			       b.bytes, psnr_a, psnr_b, delta, c_diff, d_diff, e_diff < 0 ? "-" : e_str, best_ms,
			       row_ok ? "PASS" : "FAIL");
		}
		pyrowave_d3d12_encoder_destroy(encoder);
	}

	if (debug)
		ok = report_debug_layer(ctx) == 0 && ok;

	pyrowave_d3d12_device_destroy(d3d12);
	pyrowave_device_destroy(vk);
	printf("\nC = max diff D3D12 vs Vulkan decode of the D3D12 stream, D = GPU input vs CPU input,\n"
	       "E = NV12 vs planar input (both must be 0). Overall: %s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}

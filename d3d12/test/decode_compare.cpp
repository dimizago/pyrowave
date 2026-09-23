// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Decodes identical bitstreams with the Vulkan library and the D3D12 backend and
// compares the output planes.
//
// A synthetic test image is encoded with the Vulkan encoder, packetized, and pushed
// into both decoders. The Vulkan decode is the reference. With PYROWAVE_PRECISION=2
// (the default here) both sides run FP32 throughout, so they should agree to within
// rounding; with 1 the backends store the pyramid differently (the Vulkan side keeps
// the low frequency bands in FP32), so small differences are expected.
//
// --dump DIR writes each case as a test vector (bitstream + Vulkan reference) that the
// Xbox test app, or pyrowave-d3d12-vector-test, can replay without Vulkan.
//
// Usage: pyrowave-d3d12-decode-compare [--precision N] [--debug | --gbv] [--iterations N] [--dump DIR]

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
	{ 1280, 720, false },
	{ 1366, 768, true },
	{ 3840, 2160, true },
	{ 3840, 2160, false },
	{ 640, 360, false },
};

pyrowave_cpu_buffer as_cpu_buffer(Planes &p, bool chroma_444)
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

#define VK_CHECKED(x) do { pyrowave_result r_ = (x); if (r_ != PYROWAVE_SUCCESS) { \
	fprintf(stderr, "%s failed: %d\n", #x, int(r_)); return false; } } while (0)

// Encodes the synthetic image and decodes it with the Vulkan library.
bool make_vector(pyrowave_device vk, const TestCase &tc, TestVector &vec, Planes &source)
{
	source = make_test_image(tc.width, tc.height, tc.chroma_444);
	const pyrowave_chroma_subsampling chroma =
			tc.chroma_444 ? PYROWAVE_CHROMA_SUBSAMPLING_444 : PYROWAVE_CHROMA_SUBSAMPLING_420;

	vec.width = tc.width;
	vec.height = tc.height;
	vec.chroma_444 = tc.chroma_444;
	char name[64];
	snprintf(name, sizeof(name), "%dx%d_%s", tc.width, tc.height, tc.chroma_444 ? "444" : "420");
	vec.name = name;

	// Roughly 3 bits per luma pixel.
	pyrowave_encoder_create_info enc_info = {};
	enc_info.device = vk;
	enc_info.width = tc.width;
	enc_info.height = tc.height;
	enc_info.chroma = chroma;
	pyrowave_encoder encoder = nullptr;
	VK_CHECKED(pyrowave_encoder_create(&enc_info, &encoder));

	pyrowave_rate_control rc = {};
	rc.maximum_bitstream_size = size_t(tc.width) * tc.height * 3 / 8;
	pyrowave_cpu_buffer src_buf = as_cpu_buffer(source, tc.chroma_444);
	VK_CHECKED(pyrowave_encoder_encode_cpu_synchronous(encoder, &src_buf, &rc));

	const size_t boundary = 16 * 1024;
	size_t num_packets = 0;
	VK_CHECKED(pyrowave_encoder_compute_num_packets(encoder, boundary, &num_packets));
	std::vector<pyrowave_packet> packets(num_packets);
	vec.bitstream.resize(rc.maximum_bitstream_size * 2 + 64 * 1024);
	size_t out_packets = 0;
	VK_CHECKED(pyrowave_encoder_packetize(encoder, packets.data(), boundary, &out_packets, vec.bitstream.data(),
	                                      vec.bitstream.size()));
	pyrowave_encoder_destroy(encoder);

	size_t end = 0;
	vec.packets.clear();
	for (size_t i = 0; i < out_packets; i++)
	{
		vec.packets.push_back({ packets[i].offset, packets[i].size });
		end = packets[i].offset + packets[i].size > end ? packets[i].offset + packets[i].size : end;
	}
	vec.bitstream.resize(end);

	pyrowave_decoder_create_info dec_info = {};
	dec_info.device = vk;
	dec_info.width = tc.width;
	dec_info.height = tc.height;
	dec_info.chroma = chroma;
	pyrowave_decoder decoder = nullptr;
	VK_CHECKED(pyrowave_decoder_create(&dec_info, &decoder));
	for (auto &p : vec.packets)
		VK_CHECKED(pyrowave_decoder_push_packet(decoder, vec.bitstream.data() + p.offset, p.size));
	vec.reference.allocate(tc.width, tc.height, tc.chroma_444);
	pyrowave_cpu_buffer ref_buf = as_cpu_buffer(vec.reference, tc.chroma_444);
	VK_CHECKED(pyrowave_decoder_decode_cpu_buffer_synchronous(decoder, &ref_buf));
	pyrowave_decoder_destroy(decoder);
	return true;
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
	int iterations = 20;
	bool debug = false, gpu_validation = false;
	std::wstring dump_dir;
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
		else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
		{
			std::string d = argv[++i];
			dump_dir.assign(d.begin(), d.end());
		}
	}

	// The Vulkan library reads this when its device is created.
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
	pyrowave_d3d12_result res = pyrowave_d3d12_device_create(&info, &d3d12);
	if (res != PYROWAVE_D3D12_SUCCESS)
	{
		fprintf(stderr, "pyrowave_d3d12_device_create: %s\n", pyrowave_d3d12_result_to_string(res));
		return 1;
	}

	printf("D3D12 adapter: %s, PYROWAVE_PRECISION=%d%s\n\n", ctx.adapter_name.c_str(), precision,
	       gpu_validation ? ", debug layer + GPU-based validation on" : (debug ? ", debug layer on" : ""));

	int worst = 0;
	bool ok = true;
	static const char *names[3] = { "Y ", "Cb", "Cr" };
	for (auto &tc : test_cases)
	{
		TestVector vec;
		Planes source, decoded;
		DecodeStats stats;
		std::string error;
		if (!make_vector(vk, tc, vec, source))
		{
			ok = false;
			continue;
		}
		if (!decode(ctx, d3d12, vec, iterations, decoded, stats, error))
		{
			fprintf(stderr, "%s: %s\n", vec.name.c_str(), error.c_str());
			ok = false;
			continue;
		}

		size_t bytes = 0;
		for (auto &p : vec.packets)
			bytes += p.size;
		printf("%4dx%-4d %s  %7.1f KB  D3D12 decode %.3f ms (median %.3f)\n", tc.width, tc.height,
		       tc.chroma_444 ? "444" : "420", double(bytes) / 1024.0, stats.best_ms, stats.median_ms);
		for (int i = 0; i < 3; i++)
		{
			PlaneDiff d = compare(vec.reference.data[i], decoded.data[i]);
			worst = d.max_diff > worst ? d.max_diff : worst;
			printf("    %s  vs Vulkan: max diff %3d, %8zu px differ (%6.3f%%) | PSNR vs source: Vulkan %.2f dB, D3D12 %.2f dB\n",
			       names[i], d.max_diff, d.mismatches, 100.0 * double(d.mismatches) / double(d.total),
			       psnr(source.data[i], vec.reference.data[i]), psnr(source.data[i], decoded.data[i]));
		}

		if (!dump_dir.empty())
		{
			std::wstring wname(vec.name.begin(), vec.name.end());
			std::wstring path = dump_dir + L"\\" + wname + L".pwtv";
			if (!write_test_vector(path, vec))
			{
				fprintf(stderr, "Failed to write %ls\n", path.c_str());
				ok = false;
			}
		}
	}

	if (debug)
		ok = report_debug_layer(ctx) == 0 && ok;

	pyrowave_d3d12_device_destroy(d3d12);
	pyrowave_device_destroy(vk);

	// FP32 on both sides should agree to within rounding.
	const int tolerance = precision == 2 ? 1 : 2;
	printf("\nWorst difference vs Vulkan: %d (tolerance %d) -> %s\n", worst, tolerance,
	       ok && worst <= tolerance ? "PASS" : "FAIL");
	return ok && worst <= tolerance ? 0 : 1;
}

// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Replays test vectors dumped by pyrowave-d3d12-decode-compare --dump, without Vulkan.
// The same suite runs on Xbox in test/uwp.
//
// Usage: pyrowave-d3d12-vector-test [--iterations N] [--debug | --gbv] [--r16] [--precision N]...
//                                   [--baseline-save DIR | --baseline DIR] file.pwtv...
//
// --tier-bench first runs the GPU partition micro-benchmarks (tier_bench.hpp).
// --r16 decodes into R16_UNORM planes (what the Moonlight client uses) instead of R8.
// --baseline-save DIR stores every raw output; a later --baseline DIR run (e.g. after a
// shader change) fails unless each output is bit-identical to the stored one.
// --baseline-quality DIR instead reports each output's error against the stored
// precision 2 (FP32) output of the same vector.

#include "tier_bench.hpp"
#include "vector_test.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

using namespace PyroWaveTest;

int wmain(int argc, wchar_t **argv)
{
	int iterations = 50;
	bool debug = false, gpu_validation = false, explicit_precision = false, tier_bench = false;
	std::vector<TestVector> vectors;
	for (int i = 1; i < argc; i++)
	{
		if (!wcscmp(argv[i], L"--iterations") && i + 1 < argc)
			iterations = _wtoi(argv[++i]);
		else if (!wcscmp(argv[i], L"--debug"))
			debug = true;
		else if (!wcscmp(argv[i], L"--gbv"))
			debug = gpu_validation = true;
		else if (!wcscmp(argv[i], L"--tier-bench"))
			tier_bench = true;
		else if (!wcscmp(argv[i], L"--r16"))
			g_output_format = DXGI_FORMAT_R16_UNORM;
		else if (!wcscmp(argv[i], L"--precision") && i + 1 < argc)
		{
			if (!explicit_precision)
				g_precisions.clear();
			explicit_precision = true;
			g_precisions.push_back(_wtoi(argv[++i]));
		}
		else if ((!wcscmp(argv[i], L"--baseline") || !wcscmp(argv[i], L"--baseline-save") ||
		          !wcscmp(argv[i], L"--baseline-quality")) && i + 1 < argc)
		{
			g_baseline_save = !wcscmp(argv[i], L"--baseline-save");
			g_baseline_quality = !wcscmp(argv[i], L"--baseline-quality");
			g_baseline_dir = argv[++i];
		}
		else
		{
			TestVector vec;
			if (!read_test_vector(argv[i], vec))
			{
				fprintf(stderr, "Failed to read %ls\n", argv[i]);
				return 1;
			}
			// File name without directory and extension, as the vector's name.
			std::wstring w = argv[i];
			size_t slash = w.find_last_of(L"\\/");
			w = w.substr(slash == std::wstring::npos ? 0 : slash + 1);
			w = w.substr(0, w.find_last_of(L'.'));
			for (wchar_t c : w)
				vec.name += c < 128 ? char(c) : '?';
			vectors.push_back(std::move(vec));
		}
	}

	if (vectors.empty())
	{
		fprintf(stderr, "Usage: pyrowave-d3d12-vector-test [--iterations N] [--debug | --gbv] [--r16] [--precision N]... "
		                "[--baseline-save DIR | --baseline DIR | --baseline-quality DIR] file.pwtv...\n");
		return 1;
	}

	Context ctx;
	if (!ctx.init(debug, gpu_validation))
	{
		fprintf(stderr, "Failed to create the D3D12 device.\n");
		return 1;
	}

	if (tier_bench)
		fputs(run_tier_bench(ctx).c_str(), stdout);

	bool pass = false;
	fputs(run_suite(ctx, vectors, iterations, pass).c_str(), stdout);
	return pass ? 0 : 1;
}

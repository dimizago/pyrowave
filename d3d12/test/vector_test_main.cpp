// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

// Replays test vectors dumped by pyrowave-d3d12-decode-compare --dump, without Vulkan.
// The same suite runs on Xbox in test/uwp.
//
// Usage: pyrowave-d3d12-vector-test [--iterations N] [--debug | --gbv] file.pwtv...

#include "vector_test.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

using namespace PyroWaveTest;

int wmain(int argc, wchar_t **argv)
{
	int iterations = 50;
	bool debug = false, gpu_validation = false;
	std::vector<TestVector> vectors;
	for (int i = 1; i < argc; i++)
	{
		if (!wcscmp(argv[i], L"--iterations") && i + 1 < argc)
			iterations = _wtoi(argv[++i]);
		else if (!wcscmp(argv[i], L"--debug"))
			debug = true;
		else if (!wcscmp(argv[i], L"--gbv"))
			debug = gpu_validation = true;
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
		fprintf(stderr, "Usage: pyrowave-d3d12-vector-test [--iterations N] [--debug | --gbv] file.pwtv...\n");
		return 1;
	}

	Context ctx;
	if (!ctx.init(debug, gpu_validation))
	{
		fprintf(stderr, "Failed to create the D3D12 device.\n");
		return 1;
	}

	bool pass = false;
	fputs(run_suite(ctx, vectors, iterations, pass).c_str(), stdout);
	return pass ? 0 : 1;
}

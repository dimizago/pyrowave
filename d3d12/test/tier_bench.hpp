// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT
#pragma once

// Characterises the GPU partition the process runs in (ALU latency and throughput, read
// bandwidth, typed store rate, dispatch overhead), to tell what a reduced Xbox resource
// tier actually throttles.

#include "vector_test.hpp"

#include <string>

namespace PyroWaveTest
{
std::string run_tier_bench(Context &ctx);
}

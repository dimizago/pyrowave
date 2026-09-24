# Direct3D 12 backend. Builds with MSVC and MinGW; needs no shader tools, since the
# DXIL in shaders/generated is produced by shaders/transpile.py and committed.
#
# Works as part of the PyroWave build (option PYROWAVE_D3D12) or standalone /
# add_subdirectory()'d on its own, e.g. by a streaming host: it needs nothing from the
# rest of the repository except com_ptr.hpp and metal/pyrowave_bitstream.*.
cmake_minimum_required(VERSION 3.20)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(pyrowave-d3d12 LANGUAGES CXX)
endif()

add_library(pyrowave-d3d12 STATIC
        pyrowave_d3d12.h
        pyrowave_d3d12_internal.hpp
        pyrowave_d3d12_device.cpp
        pyrowave_d3d12_decoder.cpp
        pyrowave_d3d12_encoder.cpp
        ../metal/pyrowave_bitstream.cpp ../metal/pyrowave_bitstream.hpp
        shaders/generated/pyrowave_dxil.h)
target_include_directories(pyrowave-d3d12 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(pyrowave-d3d12 PRIVATE cxx_std_14)
target_compile_definitions(pyrowave-d3d12 PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(pyrowave-d3d12 PUBLIC d3d12)

# Test harness shared by the tools below and the Xbox test app (test/uwp).
add_library(pyrowave-d3d12-test-harness STATIC test/vector_test.cpp test/vector_test.hpp
        test/tier_bench.cpp test/tier_bench.hpp test/tier_bench_dxil.h)
target_link_libraries(pyrowave-d3d12-test-harness PUBLIC pyrowave-d3d12 dxgi)
target_compile_definitions(pyrowave-d3d12-test-harness PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

# Replays dumped test vectors; needs no Vulkan.
add_executable(pyrowave-d3d12-vector-test test/vector_test_main.cpp)
target_link_libraries(pyrowave-d3d12-vector-test PRIVATE pyrowave-d3d12-test-harness)
if (MINGW)
    target_link_options(pyrowave-d3d12-vector-test PRIVATE -municode)
endif()

# Decodes the same Vulkan-encoded bitstreams with both backends and compares.
if (TARGET pyrowave-shared)
    add_executable(pyrowave-d3d12-decode-compare test/decode_compare.cpp)
    target_link_libraries(pyrowave-d3d12-decode-compare PRIVATE pyrowave-d3d12-test-harness pyrowave-shared granite-volk-headers)
    target_include_directories(pyrowave-d3d12-decode-compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

    # Encodes with both backends and cross-decodes.
    add_executable(pyrowave-d3d12-encode-compare test/encode_compare.cpp)
    target_link_libraries(pyrowave-d3d12-encode-compare PRIVATE pyrowave-d3d12-test-harness pyrowave-shared granite-volk-headers)
    target_include_directories(pyrowave-d3d12-encode-compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

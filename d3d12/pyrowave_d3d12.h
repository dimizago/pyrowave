// Copyright (c) 2026 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#ifndef PYROWAVE_D3D12_H_
#define PYROWAVE_D3D12_H_

// Direct3D 12 backend for PyroWave, without Granite or Vulkan.
//
// Targets any D3D12 device with Shader Model 6.4 and wave operations, including the
// Xbox Series UWP (Dev Mode) GPU partition, which is feature level 11_0, wave64 only,
// has no native 16-bit shader types and no typed UAV loads beyond R32. The shaders are
// written to that lowest common denominator; see shaders/transpile.py.
//
// Entry points are prefixed pyrowave_d3d12_ so this library can be linked into the
// same binary as the Vulkan library (pyrowave.h), e.g. for comparison testing.
//
// The encoder additionally needs Shader Model 6.6 with native 16-bit types and a
// 64-wide wave size (any recent desktop GPU); it is meant for the streaming host.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#if !defined(PYROWAVE_D3D12_PUBLIC_API)
#if defined(PYROWAVE_D3D12_EXPORT_SYMBOLS) && defined(_WIN32)
#define PYROWAVE_D3D12_PUBLIC_API __declspec(dllexport)
#else
#define PYROWAVE_D3D12_PUBLIC_API
#endif
#endif

// Forward declarations so this header does not pull in d3d12.h.
struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;
struct ID3D12Fence;

// Codes 0 and -1 through -4 carry the same meaning as in pyrowave.h.
typedef enum pyrowave_d3d12_result
{
	PYROWAVE_D3D12_SUCCESS = 0,
	PYROWAVE_D3D12_ERROR_GENERIC = -1,
	PYROWAVE_D3D12_ERROR_INVALID_ARGUMENT = -2,
	PYROWAVE_D3D12_ERROR_OUT_OF_HOST_MEMORY = -3,
	PYROWAVE_D3D12_ERROR_OUT_OF_DEVICE_MEMORY = -4,
	PYROWAVE_D3D12_ERROR_UNSUPPORTED_DEVICE = -5,
	PYROWAVE_D3D12_ERROR_CORRUPT_BITSTREAM = -7,
	PYROWAVE_D3D12_ERROR_INT_MAX = 0x7fffffff
} pyrowave_d3d12_result;

typedef enum pyrowave_d3d12_chroma_subsampling
{
	PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420 = 0,
	PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444 = 1,
	PYROWAVE_D3D12_CHROMA_SUBSAMPLING_INT_MAX = 0x7fffffff
} pyrowave_d3d12_chroma_subsampling;

typedef struct pyrowave_d3d12_device_opaque *pyrowave_d3d12_device;
typedef struct pyrowave_d3d12_decoder_opaque *pyrowave_d3d12_decoder;
typedef struct pyrowave_d3d12_encoder_opaque *pyrowave_d3d12_encoder;

typedef void (*pyrowave_d3d12_message_cb)(void *userdata, const char *msg);

PYROWAVE_D3D12_PUBLIC_API const char *
pyrowave_d3d12_result_to_string(pyrowave_d3d12_result result);

// Device API.

typedef struct pyrowave_d3d12_device_create_info
{
	// Required. PyroWave takes a reference for as long as it needs it.
	struct ID3D12Device *d3d12_device;

	// Optional message callback, or NULL to print to stderr / OutputDebugString.
	pyrowave_d3d12_message_cb message_callback;
	void *message_userdata;

	// Wavelet precision: 2 is FP32 throughout, 1 keeps FP32 math but stores the
	// wavelet pyramid as FP16 (faster, within 1 LSB). 0 means use the
	// PYROWAVE_PRECISION environment variable if set, else the default (1).
	int wavelet_precision;
} pyrowave_d3d12_device_create_info;

// Reports whether a device can run the decoder: Shader Model 6.4 and wave operations
// with at least 4 lanes.
PYROWAVE_D3D12_PUBLIC_API bool
pyrowave_d3d12_device_is_supported(struct ID3D12Device *d3d12_device);

// Reports whether a device can additionally run the encoder: Shader Model 6.6, native
// 16-bit shader types and support for 64-wide waves.
PYROWAVE_D3D12_PUBLIC_API bool
pyrowave_d3d12_device_supports_encoder(struct ID3D12Device *d3d12_device);

// Creates the root signature and pipelines, so create one and share it across decoders.
//
// Unless wavelet_precision is set, the PYROWAVE_PRECISION environment variable selects
// the precision as on the Vulkan side, except that 0 (FP16 math) is treated as 1, since
// native 16-bit arithmetic is not available on every target.
//
// On Xbox Series at precision 1, the decoder uses kernels written for the
// moonlight-xbox client (shaders/xbox): that client needs the hevcPlayback capability
// for HDR10 output, which confines it to the console's reduced "4K media app" GPU
// partition, and there they decode a 4K 4:4:4 frame at streaming rates in 1.8 ms
// instead of 4.5 ms, and more accurately. Other devices use the translated kernels.
// PYROWAVE_D3D12_XBOX_KERNELS=0 disables them; =1 runs them on a PC GPU with 64-lane
// wave support, for testing.
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_device_create(const pyrowave_d3d12_device_create_info *info, pyrowave_d3d12_device *device);

// All decoders created from this device must be destroyed first.
PYROWAVE_D3D12_PUBLIC_API void
pyrowave_d3d12_device_destroy(pyrowave_d3d12_device device);

// Decoder API.

typedef struct pyrowave_d3d12_decoder_create_info
{
	pyrowave_d3d12_device device;

	// Luma dimensions. For 420 subsampling both must be even.
	// Both must be in the range [1, 16384], as the bitstream encodes them in 14 bits.
	int width;
	int height;

	pyrowave_d3d12_chroma_subsampling chroma;
} pyrowave_d3d12_decoder_create_info;

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_decoder_create(const pyrowave_d3d12_decoder_create_info *info, pyrowave_d3d12_decoder *decoder);

// Waits for all of this decoder's GPU work that was reported through completion fences.
PYROWAVE_D3D12_PUBLIC_API void
pyrowave_d3d12_decoder_destroy(pyrowave_d3d12_decoder decoder);

// Throws away all queued packets.
PYROWAVE_D3D12_PUBLIC_API void
pyrowave_d3d12_decoder_clear(pyrowave_d3d12_decoder decoder);

// Same semantics as pyrowave_decoder_push_packet(). Returns
// PYROWAVE_D3D12_ERROR_CORRUPT_BITSTREAM if the data does not parse.
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_decoder_push_packet(pyrowave_d3d12_decoder decoder, const void *data, size_t size);

// Like push_packet, but for a buffer known to be cut short (the tail of a network
// frame was lost): running out of data mid-packet is not an error, and whole packets
// ahead of the cut are still decoded.
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_decoder_push_packet_truncated(pyrowave_d3d12_decoder decoder, const void *data, size_t size);

PYROWAVE_D3D12_PUBLIC_API bool
pyrowave_d3d12_decoder_decode_is_ready(pyrowave_d3d12_decoder decoder, bool allow_partial_frame);

// Readiness rule for frames that arrive as a byte prefix of the packet stream: a
// partial frame is accepted once any of its packets started past the coarse
// decomposition levels (4 and 3), which proves every transmitted coarse block
// arrived, so the result degrades to blur rather than garbage. Much more permissive
// than decode_is_ready()'s block ratio rule for tail-truncated frames.
PYROWAVE_D3D12_PUBLIC_API bool
pyrowave_d3d12_decoder_decode_is_ready_prefix(pyrowave_d3d12_decoder decoder, bool allow_partial_frame);

// Blocks decoded so far in the current frame, and how many the sequence header says
// were transmitted.
PYROWAVE_D3D12_PUBLIC_API void
pyrowave_d3d12_decoder_get_block_counts(pyrowave_d3d12_decoder decoder, int *decoded_blocks, int *total_blocks);

PYROWAVE_D3D12_PUBLIC_API bool
pyrowave_d3d12_decoder_decode_is_ready_with_sideband(pyrowave_d3d12_decoder decoder, bool allow_partial_frame,
		int num_pristine_bands, float minimum_packet_ratio,
		const uint32_t *active_block_mask, size_t word_count);

// Y, Cb, Cr as three separate single-channel 2D textures.
//
// Requirements on each texture:
//   - D3D12_RESOURCE_DIMENSION_TEXTURE2D, one mip, array size 1, no MSAA
//   - D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
//   - a single-channel format with typed UAV store support (R8_UNORM, R16_UNORM,
//     R16_FLOAT and R32_FLOAT are the expected choices; the shader writes normalized float)
//   - plane 0 is width x height; for 420 chroma, planes 1 and 2 are
//     (width / 2) x (height / 2), and for 444 they match plane 0
//   - in D3D12_RESOURCE_STATE_UNORDERED_ACCESS when the recorded work executes.
//     They are left in that state.
typedef struct pyrowave_d3d12_gpu_buffers
{
	struct ID3D12Resource *planes[3];
} pyrowave_d3d12_gpu_buffers;

// Records the decode into a compute-capable command list (DIRECT or COMPUTE). Nothing
// is submitted; the caller owns execution.
//
// The library binds its own shader-visible CBV/SRV/UAV descriptor heap on the command
// list (SetDescriptorHeaps), plus its compute root signature and pipelines. The caller
// must re-bind its own heap and root signature afterwards if it records more work.
//
// A UAV barrier on the output planes is recorded at the end, so later work in the
// same command list can read them after transitioning them as needed.
//
// completion_fence / completion_value: the caller must signal completion_fence to at
// least completion_value on the executing queue after this command list, e.g. with
// ID3D12CommandQueue::Signal(). The decoder uses it to recycle its upload buffers and
// descriptors, and blocks in a later decode (or destroy) if the caller is more than a
// few frames ahead of the GPU. It is required.
//
// Decoding may be requested at any time, producing incomplete results if packets are
// missing (missing wavelet coefficients are treated as 0, which shows up as blurring).
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_decoder_decode_gpu_buffer(pyrowave_d3d12_decoder decoder,
                                         struct ID3D12GraphicsCommandList *command_list,
                                         const pyrowave_d3d12_gpu_buffers *buffers,
                                         struct ID3D12Fence *completion_fence,
                                         uint64_t completion_value);


// Encoder API.
//
// Same model as the Metal port: the encoder owns a compute queue, an encode call
// submits GPU work and returns without blocking, and the packet queries block until
// that work has finished. Encoding again clobbers the previous frame's result. The
// bitstream is identical in format to the Vulkan encoder's.

typedef struct pyrowave_d3d12_encoder_create_info
{
	pyrowave_d3d12_device device;

	// Luma dimensions. For 420 subsampling both must be even. Both in [1, 16384].
	int width;
	int height;

	pyrowave_d3d12_chroma_subsampling chroma;
} pyrowave_d3d12_encoder_create_info;

typedef struct pyrowave_d3d12_packet
{
	size_t offset;
	size_t size;
} pyrowave_d3d12_packet;

typedef struct pyrowave_d3d12_rate_control
{
	// The bitstream for an image will not exceed this size.
	size_t maximum_bitstream_size;
} pyrowave_d3d12_rate_control;

typedef enum pyrowave_d3d12_cpu_buffer_format
{
	PYROWAVE_D3D12_CPU_BUFFER_FORMAT_NV12 = 0,    // 2 planes. Y in 8bpp, then CbCr interleaved in 16bpp.
	PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV420P = 1, // 3 planes, half resolution chroma.
	PYROWAVE_D3D12_CPU_BUFFER_FORMAT_YUV444P = 2, // 3 planes, full resolution chroma.
	PYROWAVE_D3D12_CPU_BUFFER_FORMAT_INT_MAX = 0x7fffffff
} pyrowave_d3d12_cpu_buffer_format;

typedef struct pyrowave_d3d12_cpu_buffer
{
	const void *data[3];
	// Must be at least width for plane times texel size of the plane.
	size_t row_stride_in_bytes[3];
	// Must be at least row_stride times height of plane.
	size_t plane_size_in_bytes[3];
	// Size of the luma plane; must match the encoder.
	int width;
	int height;
	pyrowave_d3d12_cpu_buffer_format format;
} pyrowave_d3d12_cpu_buffer;

// GPU input. Either:
//   - three single-channel 2D textures in planes[0..2] (Y, Cb, Cr; chroma at half
//     resolution for 420), any UNORM/FLOAT format the shader can sample (R8_UNORM,
//     R16_UNORM, R16_FLOAT, ...), or
//   - one planar DXGI_FORMAT_NV12 or DXGI_FORMAT_P010 texture in planes[0] with
//     planes[1] and planes[2] NULL (420 only). Its chroma plane is sampled twice
//     through swizzled views, so no deinterleave copy is needed.
// The planes must be readable by compute shaders when the encode executes: in
// D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE (or ALL_SHADER_RESOURCE), or in
// COMMON when the resource allows implicit promotion (e.g. simultaneous access, or
// shared with another API). They are only read.
//
// wait_fence / wait_value: if set, the encoder's queue waits on the GPU for the fence
// to reach the value before reading the planes (e.g. a fence the capture side signals
// once the frame is in place). signal_fence / signal_value: if set, signalled on the
// encoder's queue once the encode, input reads included, has finished.
typedef struct pyrowave_d3d12_encoder_gpu_input
{
	struct ID3D12Resource *planes[3];
	struct ID3D12Fence *wait_fence;
	uint64_t wait_value;
	struct ID3D12Fence *signal_fence;
	uint64_t signal_value;
} pyrowave_d3d12_encoder_gpu_input;

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_create(const pyrowave_d3d12_encoder_create_info *info, pyrowave_d3d12_encoder *encoder);

// Waits for any in-flight encode.
PYROWAVE_D3D12_PUBLIC_API void
pyrowave_d3d12_encoder_destroy(pyrowave_d3d12_encoder encoder);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_encode_gpu(pyrowave_d3d12_encoder encoder, const pyrowave_d3d12_encoder_gpu_input *input,
                                  const pyrowave_d3d12_rate_control *rate_control);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_encode_cpu(pyrowave_d3d12_encoder encoder, const pyrowave_d3d12_cpu_buffer *input,
                                  const pyrowave_d3d12_rate_control *rate_control);

// Packet queries: only valid after a successful encode, and only for that frame.
// They block until the GPU has finished it. Semantics match pyrowave.h.
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_compute_num_packets(pyrowave_d3d12_encoder encoder, size_t packet_boundary,
                                           size_t *num_packets);
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_compute_num_packets_with_padding(pyrowave_d3d12_encoder encoder, size_t packet_boundary,
                                                        size_t padding_size, size_t *num_packets);
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_compute_num_critical_packets(pyrowave_d3d12_encoder encoder, int bands,
                                                    size_t packet_boundary, size_t padding_size,
                                                    size_t *num_packets);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_packetize(pyrowave_d3d12_encoder encoder, pyrowave_d3d12_packet *packets,
                                 size_t packet_boundary, size_t *out_packets, void *bitstream, size_t size);
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_packetize_with_padding(pyrowave_d3d12_encoder encoder, pyrowave_d3d12_packet *packets,
                                              size_t packet_boundary, size_t padding_size, size_t *out_packets,
                                              void *bitstream, size_t size);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_get_mapped_raw_bitstream(pyrowave_d3d12_encoder encoder,
                                                const void **mapped_bitstream, size_t *mapped_bitstream_size,
                                                const void **mapped_metadata, size_t *mapped_metadata_size);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_get_num_active_blocks(pyrowave_d3d12_encoder encoder, int bands, size_t *num_active_blocks);

PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_compute_block_active_words(pyrowave_d3d12_encoder encoder, int bands,
                                                  uint32_t *words, size_t word_count);

// GPU time of the last encode in milliseconds (blocks until it has finished).
PYROWAVE_D3D12_PUBLIC_API pyrowave_d3d12_result
pyrowave_d3d12_encoder_get_last_gpu_time(pyrowave_d3d12_encoder encoder, double *milliseconds);

#ifdef __cplusplus
}
#endif

#endif

# PyroWave D3D12 backend

A native Direct3D 12 port of the PyroWave encoder and decoder, with no Granite or Vulkan dependency. It exists for one streaming setup:

- **Encoder:** the Vibepollo (Sunshine) host on a Windows PC.
- **Decoder:** the [moonlight-xbox](https://github.com/TheElixZammuto/moonlight-xbox) client on Xbox Series consoles, where D3D12 is the only API with wave intrinsics (Shader Model 6.4).

Most shaders are translated from `../shaders/*.comp` by `shaders/transpile.py` (GLSL → SPIR-V → HLSL → DXIL). The bitstream is identical to the Vulkan library's in both directions.

This is covered by the standard MIT license in `../LICENSE`.

## The Xbox decoder kernels (`shaders/xbox`)

The dequantizer and inverse DWT in `shaders/xbox` are hand-written for the moonlight-xbox client decoding a stream on the console. They are **not** a general-purpose improvement, and the device only picks them where they were measured to help.

### Why they exist

To switch the TV to HDR10, moonlight-xbox must declare the `hevcPlayback` restricted capability. That makes it a "4K media app", which runs in a reduced GPU partition no matter what app type Dev Home is set to. Measured in that partition on a Series X (`test/tier_bench`):

| resource | measured in the partition | Series X peak |
|---|---|---|
| FP32 FMA throughput | 0.97 TFLOPS (~8%) | 12.15 TFLOPS |
| memory read/write | ~116 GB/s (~20%) | 560 GB/s |
| dependent-instruction latency | ~5x | — |
| LDS throughput | cut about as hard as ALU | — |
| barrier + dispatch | 2.8 µs | — |

Typed stores cost what the footprint of each instruction's 64 texels decides. For 8.2 M 16-bit values:

| store footprint | time |
|---|---|
| each lane writing its own 4x4 area | 1351 µs |
| one 8x8 square per instruction | 161 µs |

The translated kernels needed **4.5 ms** for a 4K 4:4:4 frame at ~460 KB, which is 4K60 at ~220 Mbps, the rate moonlight-xbox streams at. What the Xbox kernels change, and why:

- **Dequant stores one 8x8 square per instruction.** The tile is staged in LDS first. The translated kernel's per-thread 4x2 stores alone cost 2.1 ms in the partition.
- **Dequant decodes the bit planes with an 8x8 bit transpose.** One 12-byte load replaces a dependent byte load per plane.
- **The iDWT uses 64x56 tiles with 32- and 28-sample lifting segments.** The translated kernel lifts 8-sample segments, which redoes most of the apron work.
- **LDS is kept small and packed, because LDS and occupancy were the limit.**
  - The band window is a raw FP16 gather through an `R16_UINT` view of the typeless pyramid.
  - The row-to-column exchange is FP16.
  - Output is staged as the 16-bit values being stored.
- **The CDF 9/7 input scaling is folded into the dequant.**
- **Empty blocks are skipped.** At that rate 99.6% of the finest level's 32x32 blocks, and ~80% of the next level's, carry no coefficients. For those two levels the dequant doesn't write empty blocks, and the iDWT looks their emptiness up instead of reading their texels.
- **Every FP16 store rounds to nearest even first.** The hardware conversion truncates, and that truncation was the largest error in the translated decoder (see Quality).
- **The decoder schedules the work to match.** It interleaves the dequant of finer levels with the latency-bound coarse iDWT levels, and uses a single end-of-frame barrier.

### When they are used

They run only at precision 1 (moonlight-xbox's default), on devices whose waves are always 64 lanes, which among D3D12 targets means Xbox Series. On a desktop RDNA3 GPU they measured no faster, and slower at 1080p. GPUs without 64-lane waves cannot run them at all. Everywhere else the translated kernels run, unchanged bit for bit.

`PYROWAVE_D3D12_XBOX_KERNELS=0` disables them. `=1` runs them on a PC GPU with 64-lane wave support (Shader Model 6.6, `[WaveSize(64)]`) for testing.

### Decode time on Xbox Series X

All times use 16-bit output planes as in moonlight-xbox. "Translated" and "Xbox" are the two sets of kernels in the same build.

| frame | media-app partition (`hevcPlayback`): translated | media-app partition: **Xbox** | full Game-tier GPU: translated | full Game-tier GPU: **Xbox** |
|---|---|---|---|---|
| 4K 4:4:4, 460 KB | 4.49 ms | **1.78 ms** | 0.98 ms | **0.46 ms** |
| 4K 4:4:4, 3 MB | 4.51 ms | **2.29 ms** | 0.99 ms | **0.57 ms** |
| 4K 4:2:0, 460 KB | 2.31 ms | **1.06 ms** | 0.51 ms | **0.27 ms** |
| 4K 4:2:0, 3 MB | 2.35 ms | **1.50 ms** | 0.52 ms | **0.37 ms** |
| 1080p 4:2:0, 120 KB | 0.67 ms | **0.51 ms** | 0.14 ms | **0.11 ms** |

Inside moonlight-xbox itself (the in-app self-test in the `hevcPlayback` partition), the 4K 4:4:4 460 KB frame decodes in 1.76 ms.

## Quality

The test content is a 4K screenshot from a game (THE FINALS), converted to 8-bit BT.709 limited-range YCbCr. Rates match moonlight-xbox streaming: 460 KB per 4K frame is 4K60 at ~220 Mbps. PSNR is 20·log10(peak/RMS error). SSIM is Wang et al. 2004: 11x11 Gaussian window, σ 1.5, averaged over all window positions. The "avg" columns weight the planes (6·Y + Cb + Cr) / 8.

### 1. Same results as the Vulkan reference implementation

**Encoder.** Both streams are decoded by the Vulkan decoder, and both sides run at precision 1.

| frame, budget | Vulkan PSNR avg | D3D12 PSNR avg | Δ | Vulkan SSIM avg | D3D12 SSIM avg |
|---|---|---|---|---|---|
| 4K 4:4:4, 460 KB | 33.08 dB | 33.07 dB | −0.01 | 0.9154 | 0.9153 |
| 4K 4:4:4, 920 KB | 36.20 dB | 36.19 dB | −0.01 | 0.9466 | 0.9466 |
| 4K 4:4:4, 3.1 MB | 44.56 dB | 44.50 dB | −0.05 | 0.9890 | 0.9889 |
| 4K 4:2:0, 460 KB | 32.87 dB | 32.85 dB | −0.02 | 0.9118 | 0.9117 |
| 1080p 4:4:4, 230 KB | 34.80 dB | 34.82 dB | +0.02 | 0.9343 | 0.9346 |
| 1080p 4:2:0, 120 KB | 32.21 dB | 32.20 dB | −0.00 | 0.9018 | 0.9018 |

Also checked for each case:

- Every D3D12 stream stays within its byte budget.
- The D3D12 and Vulkan decodes of a D3D12 stream agree to 1 LSB.
- GPU-texture input and CPU input (planar or NV12) produce identical streams.

**Decoder.** The stream is Vulkan-encoded and compared against the Vulkan decode at precision 1. Results below use the Xbox kernels; the translated kernels are within 0.07 dB of these.

| frame | max diff vs Vulkan | pixels differing (Y) | Y PSNR / SSIM, Vulkan | Y PSNR / SSIM, D3D12 | Cb (Vulkan = D3D12) | Cr (Vulkan = D3D12) |
|---|---|---|---|---|---|---|
| 4K 4:4:4, 460 KB | 1 | 3.1% | 31.12 dB / 0.9032 | **31.12 dB / 0.9032** | 37.37 dB / 0.9400 | 40.59 dB / 0.9643 |
| 4K 4:4:4, 3 MB | 1 | 2.5% | 43.74 dB / 0.9892 | **43.74 dB / 0.9892** | 45.58 dB / 0.9859 | 48.44 dB / 0.9908 |
| 4K 4:2:0, 460 KB | 1 | 3.1% | 31.15 dB / 0.9046 | **31.15 dB / 0.9046** | 36.32 dB / 0.9206 | 39.75 dB / 0.9463 |
| 1080p 4:2:0, 120 KB | 1 | 3.4% | 30.73 dB / 0.8974 | **30.73 dB / 0.8974** | 35.13 dB / 0.9043 | 38.16 dB / 0.9259 |

All seven built-in sizes (640x360 to 4K, 4:2:0 and 4:4:4, including odd band sizes) are within 1 LSB of Vulkan with the D3D12 debug layer clean. On the console, the Xbox test app checks every vector against the Vulkan reference as well.

### 2. Decoder precision

Precision 1 stores the wavelet pyramid as FP16. Measured against the same stream decoded entirely in FP32 (precision 2), on the 16-bit output planes moonlight-xbox renders from (errors in 16-bit units):

| frame | translated kernels: max / RMS / PSNR | Xbox kernels: max / RMS / PSNR |
|---|---|---|
| 4K 4:4:4, 460 KB | 168 / 31.6 / 66.3 dB | **91 / 5.8 / 81.1 dB** |
| 4K 4:4:4, 3 MB | 179 / 34.8 / 65.5 dB | **79 / 5.7 / 81.2 dB** |
| 4K 4:2:0, 460 KB | 172 / 42.5 / 63.8 dB | **81 / 7.7 / 78.7 dB** |
| 4K 4:2:0, 3 MB | 177 / 46.4 / 63.0 dB | **79 / 7.6 / 78.7 dB** |
| 1080p 4:2:0, 120 KB | 165 / 42.4 / 63.8 dB | **74 / 7.8 / 78.5 dB** |

The translated decoder's error came mostly from the hardware truncating on every float → FP16 store. The Xbox kernels round to nearest even first. Their worst case is about 1.4 LSB of a 10-bit signal, against 3 LSB before.

### 3. What these numbers mean

Section 1 shows both D3D12 sides reproduce the reference codec: same PSNR and SSIM against the source, to within a few hundredths of a dB. Section 2 shows the decoder adds almost nothing on top of an FP32 decode.

The absolute PSNR/SSIM against the source is PyroWave itself at that rate: an intra-only wavelet codec at 0.4 bits per pixel. That is the trade it makes for sub-millisecond encode and decode. It is not something the D3D12 port changes.

## Reproducing

The tools are built with the rest of PyroWave (CMake preset `msvc-release`). `decode-compare` and `encode-compare` also need the Vulkan library.

```
# Real frame -> raw planar YUV (ffmpeg)
ffmpeg -i shot.jpg -vf scale=out_color_matrix=bt709:out_range=tv -pix_fmt yuv444p -f rawvideo game_3840x2160_444.yuv

# Encoder vs Vulkan, with PSNR/SSIM vs the source
pyrowave-d3d12-encode-compare --precision 1 --only 3840x2160_444 --source game_3840x2160_444.yuv --bytes 460000

# Decoder vs Vulkan (add PYROWAVE_D3D12_XBOX_KERNELS=1 to run the Xbox kernels on a PC GPU);
# --dump writes a test vector for the tools below and the Xbox test app
pyrowave-d3d12-decode-compare --precision 1 --only 3840x2160_444 --source game_3840x2160_444.yuv --bytes 460000 --dump vectors

# Precision vs FP32: save FP32 outputs, then measure precision 1 against them
pyrowave-d3d12-vector-test --r16 --precision 2 --baseline-save baseline vectors\*.pwtv
pyrowave-d3d12-vector-test --r16 --precision 1 --baseline-quality baseline vectors\*.pwtv

# GPU partition characteristics
pyrowave-d3d12-vector-test --tier-bench vectors\*.pwtv
```

On the console, `test/uwp` is a UWP test app that declares `hevcPlayback` like moonlight-xbox. It replays the vectors in `test/uwp/vectors` with both sets of kernels and writes its report to `LocalState\pyrowave_d3d12_test.txt`. Swapping the capability for `expandedResources` gives it the Game-tier GPU instead.

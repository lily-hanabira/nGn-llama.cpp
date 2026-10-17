# Spec: x86 SIMD kernel for Q2_0 (AVX2, optional AVX-512)

Status: implemented 2026-10-17 on top of `0438615d6` (AVX2, AVX-512 VNNI and AVX-VNNI paths);
see [Results](#results).

## Why

PrismML ternary models (`Ternary-Bonsai-2-4B/9B-PQ2_0-MTP-Q8_0.gguf`) are stored as PQ2_0 and
expanded to Q2_0 at load (`src/llama-model-loader.cpp`, `llama_expand_pq2_0`). On x86 the
Q2_0 dot product is the scalar fallback: `ggml/src/ggml-cpu/arch-fallback.h` maps
`ggml_vec_dot_q2_0_q8_0_generic` to `ggml_vec_dot_q2_0_q8_0` in the x86 section, and
`arch/x86/quants.c` has no Q2_0 function. Only ARM has a vectorised kernel
(`arch/arm/quants.c`, NEON with `vdotq`).

Measured on an AMD Ryzen 9 7945HX (Zen 4), CPU only (`--device none -ngl 0`, CUDA and Vulkan
hidden), 4 threads pinned to one hardware thread per physical core (affinity `0x55`):
Ternary-Bonsai-2-4B decodes at **1.26 tokens/s**. The same model on an RTX 5070 decodes at
about 200 tokens/s. CPU inference with these models is unusable until this kernel exists.

## Data layout (from `ggml/src/ggml-common.h`)

```c
#define QK2_0 64
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;   // 18 bytes, 64 weights
#define QK8_0 32
typedef struct { ggml_half d; int8_t  qs[QK8_0];     } block_q8_0;   // 34 bytes, 32 values
```

- `vec_dot_type` for Q2_0 is Q8_0 (`ggml-cpu.c` type traits, `nrows = 1`). One Q2_0 block
  pairs with two consecutive Q8_0 blocks: `y[2*i + k]`, k = 0, 1.
- Half k of block i uses `x[i].qs[8*k .. 8*k + 7]` against `y[2*i + k].qs[0 .. 31]`.
- Codes are **interleaved within each byte**, low bits first: weight `4*b + m` of a half is
  `(qs[b] >> (2*m)) & 3`, m = 0..3. Codes map {0, 1, 2, 3} to {-1, 0, +1, +2}, so the
  weight value is `code - 1`.
- Result: `sum_i d0(x_i) * sum_k d1(y_{2i+k}) * sum_j (code_j - 1) * y_j`.

Reference implementations to match: `ggml_vec_dot_q2_0_q8_0_generic` (`ggml-cpu/quants.c`,
around line 177) and the NEON kernel (`arch/arm/quants.c`, around line 222).

## Deliverable

1. `ggml_vec_dot_q2_0_q8_0` in `ggml/src/ggml-cpu/arch/x86/quants.c` with an AVX2 path, and
   the scalar fallback when AVX2 is unavailable (same pattern as `ggml_vec_dot_q1_0_q8_0` at
   about line 555 of that file).
2. Remove the x86 `#define ggml_vec_dot_q2_0_q8_0_generic ggml_vec_dot_q2_0_q8_0` from
   `arch-fallback.h` (the x86 section, about line 90) so the x86 implementation is used.
3. Optional, same change or a follow-up: an AVX-512 VNNI path (`_mm512_dpbusd_epi32` or
   `_mm256_dpbusd_epi32` with AVX-VNNI), selected by `__AVX512VNNI__` / `__AVXVNNI__`.

Keep the signature, `assert(nrc == 1)` and the `UNUSED` lines as in the other kernels.

## Suggested AVX2 approach (implementer may choose another)

Per block, per half (8 code bytes, 32 weights):

1. **Unpack to 32 unsigned codes in weight order.** One way: broadcast the 8 code bytes
   into each 64-bit lane (`_mm256_set1_epi64x`), shift lanes right by {0, 2, 4, 6} with
   `_mm256_srlv_epi64`, AND each byte with 0x03. That yields codes grouped by m (planar).
   Either reorder them to weight order with `_mm256_shuffle_epi8` plus
   `_mm256_permutevar8x32_epi32`, or (cheaper) reorder the Q8_0 values instead once per half:
   the dot product only needs x and y in the same order. Alternatively, replicate each byte
   four times with `_mm256_shuffle_epi8`, as the NEON kernel does with `vqtbl1q`, and
   emulate the per-byte variable shift with a multiply or a per-position mask plus a small
   lookup table.
2. **Dot product.** Codes are unsigned 0..3 and Q8_0 values are signed, which suits
   `_mm256_maddubs_epi16(codes, y)` followed by `_mm256_madd_epi16(…, ones)`, giving
   `sum(code * y)` in int32 lanes. Subtract `sum(y)` for the `- 1`: compute it with
   `_mm256_maddubs_epi16(ones8, y)` (or fold it into the same pass). Alternatively subtract 1
   first and use the existing `mul_sum_i8_pairs_float` sign trick, as the Q4_0 kernel does.
   With codes at most 3 and |y| at most 127, `maddubs` cannot saturate (2 * 3 * 127 = 762).
3. **Scale and accumulate.** Convert to float and fused-multiply-add with `d0 * d1` into a
   `__m256` accumulator (`_mm256_fmadd_ps`), as `ggml_vec_dot_q4_0_q8_0` does; reduce with
   `hsum_float_8` at the end. Process both halves of a block, and ideally two blocks per
   iteration, to hide latency.

Helpers already in `arch/x86/quants.c`: `hsum_float_8`, `sum_i16_pairs_float`,
`mul_sum_us8_pairs_float`, `mul_sum_i8_pairs_float`, `bytes_from_bits_32`.

## Correctness

- Integer partial sums must equal the scalar ones exactly; only float accumulation order may
  differ.
- `tests/test-quantize-fns` must pass for `q2_0` (it already uses the ternary thresholds,
  `MAX_QUANTIZATION_TOTAL_ERROR_TERNARY` and the ternary dot-product threshold).
- Add a focused test (in `test-quantize-fns` or a new small test) comparing the SIMD result
  with `ggml_vec_dot_q2_0_q8_0_generic` on random data and on edge cases: all codes 0, all
  codes 3, y all +127, y all -128, n = 64 (one block) and n = 4096, relative error at most
  1e-5.
- `test-backend-ops -o MUL_MAT` with Q2_0 weights on the CPU backend must pass.
- Builds: MSVC 19.29 (VS 2019 Build Tools, used for the release build in
  `%TEMP%\ngn-ngn-release`, Ninja, `GGML_BACKEND_DL=ON`) and clang or gcc on Linux. Check
  `-DGGML_NATIVE=OFF -DGGML_AVX2=ON` (AVX2 only), a native Zen 4 build, and a build without
  AVX2 (fallback still compiles and passes). If CPU variants are built
  (`GGML_CPU_ALL_VARIANTS`), each x86 variant must compile.

## Performance acceptance

Measure on the same machine and settings as the baseline. The release build directory only
contains `llama-server`; build the tools first, for example
`cmake --build <build-dir> --target llama-bench test-quantize-fns test-quantize-perf`
(tests need `-DLLAMA_BUILD_TESTS=ON`). The baseline above was measured with `llama-server`
(`--device none -ngl 0 -t 4 -tb 4 --spec-draft-threads 4 --spec-type draft-mtp`, affinity set
on the process) and its `print_timing` lines; `llama-bench` numbers are preferred for the
before/after comparison.

```cmd
set CUDA_VISIBLE_DEVICES=-1
set GGML_VK_VISIBLE_DEVICES=-1
llama-bench -m Ternary-Bonsai-2-4B-PQ2_0-MTP-Q8_0.gguf -dev none -ngl 0 -t 4 -p 512 -n 128
```

Pin to one thread per physical core (process affinity `0x55` on the 7945HX, for example
`start /affinity 55 …`), then repeat with `-t 16`. Report pp512 and tg128 for the 4B and 9B
models, before and after, plus `test-quantize-perf` cycles per value for `q2_0`.

Targets for the 4B model, 4 pinned threads:

| | Baseline (scalar) | Target |
| --- | --- | --- |
| tg128 (decode) | 1.26 t/s | at least 10 t/s |
| pp512 (prompt) | to measure | at least 10 times the baseline |

The decode target is conservative: the 4B weights are about 1.1 GB as Q2_0, so decode
should approach the memory-bandwidth limit once the kernel is no longer compute-bound.

## Out of scope, possible follow-ups

- Repacked `gemv`/`gemm` for Q2_0 (as exists for Q1_0, `ggml_gemv_q1_0_4x4_q8_0`) and a
  llamafile sgemm path, both mainly for prompt processing.
- Changes to the PQ2_0 to Q2_0 expansion, CUDA, Vulkan or ARM kernels.
- `nrc == 2` (two-row) kernels.

## Results

Implementation: `ggml_vec_dot_q2_0_q8_0` in `arch/x86/quants.c`. Codes are unpacked planar
(`vpshufb` + `vpsrlvd` + mask) and the Q8_0 values are permuted to match within each 128-bit
lane, so there is no cross-lane shuffle. The `- 1` is applied as `sum(code * y) - sum(y)` with
unsigned codes, which is exact for every int8 `y` including -128 (the `sign_epi8` trick used by
`mul_sum_i8_pairs_float` is not). VNNI builds use `dpbusd` for both sums.

Correctness (MSVC 19.29, static builds, on the 7945HX):

| Build | Path exercised | `test-quantize-fns` incl. new exact test |
| --- | --- | --- |
| `/arch:AVX` (no AVX2) | scalar fallback | pass |
| `/arch:AVX2` | AVX2 `maddubs` | pass, rel. error <= 2e-8 |
| `/arch:AVX512` + VNNI | AVX-512 VNNI `dpbusd` | pass, rel. error <= 2e-8 |
| `/arch:AVX2` + AVX-VNNI | AVX-VNNI | compiles; Zen 4 lacks AVX-VNNI, not run |

- `test-backend-ops` against CUDA (RTX 5070, CPU in AVX-512 VNNI mode), `type_a=q2_0`:
  `MUL_MAT` 82 OK / 0 failed, `MUL_MAT_ID` 75 OK / 0 failed. Vulkan does not support Q2_0.
- Release-style build (`GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`): all nine x86 variants
  compile; the 7945HX loads `ggml-cpu-cascadelake`.
- gcc 14.2 (Debian trixie) and clang 22 compile the file with `-Wall -Wextra` and no new warnings
  for AVX2, `znver2`, `znver4`, AVX-VNNI, AVX and SSE4.2. Not run on Linux: no g++/cmake there.

`test-quantize-perf --type q2_0 --op vec_dot_q -4` (MSVC reports 0 cycles, so throughput is used),
quantized GB/s, one thread:

| Working set | Scalar | AVX2 | AVX-512 VNNI |
| --- | ---: | ---: | ---: |
| L1 | 0.77 | 7.15 | 7.15 |
| L2 | 0.72 | 9.81 | 10.73 |
| L3 | 0.62 | 9.64 | 10.25 |
| Memory | 0.63 | 4.41 | 4.51 |

`llama-bench -dev none -ngl 0 -p 512 -n 128`, release-style build, one thread per physical core
(affinity `0x55` for 4 threads, `0x55555555` for 16), t/s:

| Model | Threads | pp512 before | pp512 after | tg128 before | tg128 after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B | 4 | 2.48 | **30.04** (12.1x) | 1.83 | **13.50** (7.4x) |
| 4B | 16 | 8.74 | 82.95 (9.5x) | 4.08 | 10.06 (2.5x) |
| 9B | 4 | 1.26 | **16.30** (12.9x) | 1.06 | **8.83** (8.3x) |
| 9B | 16 | 4.56 | 48.39 (10.6x) | 2.72 | 7.55 (2.8x) |

Both targets for the 4B model at 4 pinned threads are met (tg128 >= 10 t/s, pp512 >= 10x).
Decode no longer scales past about 4 threads on this laptop: it is bound by memory bandwidth and
thread synchronisation, so 4-8 threads are the better choice for decode.

## Handoff notes

- Context: Bastion's Nucleus agent (`bastion-nucleus/docs/engine-poc.md`, "Declared tools,
  done gate and model comparison") tested these models CPU-only and found them unusable on
  x86 for this reason.
- Keep commits small and prefix nGn-specific messages as the fork does (`... (nGn)`).

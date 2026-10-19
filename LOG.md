### Extreme FFT log

Written by Claude (an AI model by Anthropic), October 2026. Each step from the AI FFT ([fft/fft_ai.odin](https://github.com/dsego/fft-fun/blob/main/fft/fft_ai.odin) in fft-fun) to the extreme FFT and then in this repository, with what each one measured, the ones that made it slower too.

The steps up to "Two arrays" were made in fft-fun, where the extreme FFT was `fft/fft_extreme.odin`, timed whole with its `benchmark_ai` and pass by pass with a `benchmark_stages` package since removed. The ones after with this repository's [benchmark](benchmark/bench.odin).

Apple M1 Pro, Odin `dev-2026-10:84bc3fc21`, `-o:speed -microarch:native`. Real input, N = 4096 unless noted, microseconds per transform, the fastest of 5 runs. Runs of the same code differ by up to 2–3%.


#### Start

| | µs |
|---|---|
| radix-2 FFT (fft-fun's fft.odin) | 25.0 |
| AI FFT | 3.45 |
| AI FFT, fused multiply-adds in the complex multiply | 3.18 |
| pffft ordered | 2.55 |
| pffft unordered | 2.30 |

Odin doesn't fuse a multiply and an add on its own, `simd.fma` asks for it.


#### Radix-8

All stages radix-8, one radix-4 or radix-2 stage last.

| | µs |
|---|---|
| radix-8, twiddles held for the whole inner loop, 13 vector spills to the stack | 3.33 |
| radix-8, twiddles loaded in the inner loop, 2 spills | 3.43 |

One radix-8 pass cost 0.63–0.74 µs, two radix-4 passes 0.62. Fewer spills didn't help either. Dropped.


#### Passes of the AI FFT

| pass | µs |
|---|---|
| first, stride 1 | 0.62 |
| radix-4, stride 4 | 0.51 |
| radix-4, stride 16 | 0.35 |
| radix-4, stride 64 | 0.32 |
| radix-4, stride 256 | 0.31 |
| radix-2, stride 1024 | 0.24 |
| split | 0.86 |

A radix-4 pass is 128 iterations of 28 vector operations, about 3,600, and the M1 does 4 a cycle: 0.28 µs. The middle passes are close to that. The first pass, the stride 4 pass and the split aren't.


#### Split

| | µs |
|---|---|
| bin k and bin N/2 − k from the same loads (still radix-8) | 3.07 |
| back to radix-4, the last stage run inside the split | 2.84 |

The split and the last stage together: 0.76 µs, against 0.24 + 0.86 apart. 8192: 1.67 against 0.62 + 1.75.


#### Twiddles and the inner loop

| | µs |
|---|---|
| twiddles stored in all four lanes, p = 0 without multiplies, twiddles loaded for every q | 3.15 |
| the same, twiddles loaded once for each p | 2.83 |
| the p = 0 test out of the inner loop, the inner loop unrolled by two, twiddles passed as an array | 3.03 |
| back to the AI FFT's inner loop, p = 0 in a loop of its own, twiddles as six separate vectors | 2.75 |

The unrolled version slowed the middle passes to 0.49, 0.46 and 0.42 µs from 0.35, 0.32 and 0.31. The `[6]` array of twiddles, passed into an inlined procedure, is the likely cause. The last row has them at 0.34, 0.30 and 0.27, p = 0 without multiplies saving most in the late passes.


#### Instructions

| | µs |
|---|---|
| complex multiply as `fma(-x_im, w_im, x_re * w_re)`: 16 negates down to 5, folded into 11 `fmls` | 2.70 |
| first pass loads and stores in the shape LLVM turns into `ld2` and `st4` | 2.67–2.73 |

`ld2` and `st4` made no measurable difference, the first pass stayed at 0.60 µs.


#### Stride 4

At stride 4 the inner loop runs once for each p, six broadcast twiddle loads for eight data loads.

| | stride 4 pass | whole |
|---|---|---|
| before | 0.51 | 2.70 |
| twiddles of four p in six vector loads, each p's taken from its lane | 0.32 | 2.56–2.63 |
| the same at stride 16, with a loop over q | 1.13 (stride 16: 0.52) | 3.03–3.06 |

The stride 16 version was reverted.


#### Split, second look

Instructions per iteration of each loop, from `-build-mode:asm`:

| loop | before | after |
|---|---|---|
| first pass | 52 | 52 |
| radix-4 pass | 49 | 48 |
| radix-2 last stage + split | about 60 | 47 |
| radix-4 last stage + split | 244 | 191 |

The split loops reloaded the twiddle pointer and `plan.half` from the plan on every iteration, the radix-4 one the plan pointer from the stack too, 18 loads. The compiler can't rule out the stores through `out` changing the plan. Taken into locals before the loop.

The other changes to the split:

* The mirrored bins turned around in the shuffle that pairs re with im, `(3, 7, 2, 6)` and `(1, 5, 0, 4)`, instead of a reverse of each (`rev64` and `ext`) and then the pairing.
* `-odd_re` computed directly, the ½ in two multiply-adds, no negate.
* The split twiddles only up to N/4: the split pairs k with N/2 − k, k never goes past N/4. Half the table.
* The scalar edge bins by mask and shift instead of `%` and `/`.

| | 1024 | 4096 | 8192 |
|---|---|---|---|
| last stage + split, before | 0.187 | 0.712 | 1.594 |
| last stage + split, after | 0.165 | 0.623 | 1.441 |
| whole, after | 0.539–0.555 | 2.44–2.45 | 5.37–5.47 |
| pffft ordered, same runs | 0.550–0.562 | 2.51–2.54 | 5.77–5.83 |

| pass, 4096 | AI FFT | extreme FFT |
|---|---|---|
| first, stride 1 | 0.602 | 0.603 |
| radix-4, stride 4 | 0.494 | 0.320 |
| radix-4, stride 16 | 0.338 | 0.321 |
| radix-4, stride 64 | 0.309 | 0.291 |
| radix-4, stride 256 | 0.301 | 0.266 |
| last stage and split | 0.227 + 0.838 | 0.623 |
| whole | 3.138 | 2.427 |


#### First pass

The first pass with one ingredient taken out at a time, wrong output, timing only:

| first pass, N = 4096 | µs |
|---|---|
| as it is | 0.609 |
| plain stores instead of the transposing `st4` | 0.408 |
| plain loads instead of `ld2` | 0.523 |
| plain loads and stores | 0.317 |
| no twiddle loads | 0.487 |
| no twiddle multiplies | 0.448 |
| a middle pass, stride 16, for comparison | 0.322 |
| the six twiddle vectors of each block of four p next to each other | 0.602–0.619 |
| the samples 0, 8, 16 or 32 bytes past a 64 byte boundary | 0.602–0.608 |

Without its reshuffling the first pass costs what a middle pass does. The vector units (4 operations a cycle on an M1) are the limit in every pass: a middle pass is 28 vector float operations an iteration, 7 cycles, measured 7.6. The first pass adds `ld2` and `st4`, which take several vector operations each, measured 15. Loads, stores and address arithmetic run elsewhere and cost next to nothing, which is also why radix-8 and the twiddle layouts didn't help.

One reshuffle is needed somewhere: the samples come as even and odd pairs, and the stride 1 pass's outputs go to the next pass transposed. pffft has it in its last pass, the four interleaved sub-transforms combined, and its ordered transform adds a reorder on top.


#### Apple's vDSP

Accelerate's `vDSP_fft_zrip` and `vDSP_DFT_zrop`, the even and odd split of the samples (`vDSP_ctoz`) timed with them. Both return twice the DFT in two arrays, so they skip the interleave and the halving our split does.

| N | extreme FFT | pffft ordered | vDSP FFT | vDSP DFT |
|---|---|---|---|---|
| 1024 | 0.538 | 0.549 | 0.498 | 0.489 |
| 4096 | 2.401 | 2.487 | 2.291 | 2.311 |
| 8192 | 5.465 | 5.716 | 4.980 | 5.109 |


#### What limits a middle pass

Each tried on its own, N = 4096:

| | µs |
|---|---|
| radix-4 pass, stride 64, as it is | 0.297 |
| the same pass's 8 loads and 8 stores, no arithmetic | 0.246 |
| its 8 loads only | 0.140 |
| its 8 stores only | 0.223 |
| decimation in time with the twiddles in tangent form, 22 vector operations instead of 28 (correct to 8e-8) | 0.280 |
| re and im side by side in one array, `[re×4][im×4]`, paired loads and stores | 0.307 |
| the inner loop unrolled by two | 0.352 |
| a radix-8 pass's loads and stores, arrays apart, 16 + 16 streams | 0.260 |
| the same, side by side, 8 + 8 streams | 0.200 |

Memory traffic (about 0.21–0.25 µs) and arithmetic (about 0.28 µs) are close to each other, so taking away either one alone gains little: tangent twiddles cut the arithmetic 21% and the pass 3–6%, the side by side layout cut the stores' cost but not the pass's. The number of memory streams doesn't matter, a radix-8 pass's loads and stores cost what a radix-4 pass's do for half again the work.

Other measurements:

| | |
|---|---|
| a radix-4 pass in 8 lanes of `f16` instead of 4 of `f32` | 0.155 µs against 0.292, about 1e-3 precision instead of 1e-7 |
| the radix-2 split storing the bins as two arrays like vDSP | 0.549 µs against 0.610 |
| the split's scalar edge bins on their own | 0.036 µs at 4096, 0.104 at 8192, no faster inlined |
| two cores handing a flag back and forth | 121 ns a round trip |


#### Radix-8 and tangent twiddles together

A separate file, removed after these measurements: decimation in time, so the twiddles multiply the inputs and take the tangent form, radix-8 passes made of two tangent radix-4 butterflies, the first pass with no twiddles and no transpose, the last radix-4 pass reading four ways interleaved (`ld4`) inside the split. N = 4096 in 4 passes instead of 6. Correct to the same 7e-8.

| pass, N = 4096 | µs |
|---|---|
| first, radix-8, stride 256, no twiddles | 0.455 |
| radix-8, stride 32 | 0.460 |
| radix-8, stride 4, constants four p at a time | 0.422 |
| radix-4, stride 1, inside the split | 1.096 |
| whole | 2.392, the extreme FFT 2.442 |

A radix-8 pass costs about 1.5 radix-4 passes, the same per level. The inner loop shows why: 34 `mov` next to 34 `fmla` and 34 `fmls`. NEON's multiply-add overwrites its accumulator, and the tangent form adds a value into two outputs nearly everywhere (a ± ..., b' ± ...), so each one is copied first. That makes a tangent radix-4 butterfly about 33 instructions against the plain one's 28. Radix-8 with plain twiddles is 84 vector operations for 32 points over 3 levels, the same 28 per level as radix-4.


#### Two arrays

`run_arrays`: the bins into a real and an imaginary array like vDSP's, DC in `re[0]` and Nyquist in `im[0]`, no pairing of re with im in the split. Three runs each:

| N | `run` | `run_arrays` | radix-8 and tangent (removed) | pffft ordered | vDSP FFT |
|---|---|---|---|---|---|
| 1024 | 0.549–0.563 | 0.541–0.554 | 0.695–0.708 | 0.553–0.573 | 0.508–0.513 |
| 4096 | 2.448–2.501 | 2.422–2.434 | 2.414–2.442 | 2.547–2.580 | 2.341–2.381 |
| 8192 | 5.516–5.694 | 5.456–5.649 | 5.533–5.640 | 5.892–5.919 | 5.153–5.229 |

1–3% off `run`. vDSP stays 3–6% ahead.


#### Larger sizes

From here in this repository. strobe-tuner's pffft build (`clang -O2`) against `run`, two runs, the bins the same to 7–8e-8:

| N | `run` | pffft ordered | pffft unordered |
|---|---|---|---|
| 4096 | 2.48–2.53 | 2.54 | 2.28–2.29 |
| 8192 | 5.31–5.36 | 5.76–5.79 | 5.33–5.37 |
| 16384 | 12.81 | 13.17–13.20 | 11.73 |
| 32768 | 32.41–32.55 | 39.68–40.17 | 35.53–35.63 |
| 65536 | 76.08–76.11 | 78.35–78.87 | 67.16–67.18 |

vDSP pulls ahead from 16384, 10.7 µs there and 26.9 at 32768. A single transform's two work buffers and its samples outgrow the M1's 128 KB data cache past 8192.


#### Four at once

four.odin, since renamed [lanes.odin](lanes.odin) with `run4` as `run_lanes`: four transforms, one in each lane. A twiddle is the same in all four, so every pass is plain arithmetic on whole vectors: no even and odd split, no transpose, no turning lanes around in the split, and no edge bins one at a time. A transform:

| N | `run` | `run4` | vDSP FFT | vDSP DFT | vDSP FFT, 4 at once |
|---|---|---|---|---|---|
| 512 | 0.339 | 0.172 | 0.243 | 0.234 | 0.248 |
| 1024 | 0.538 | 0.436 | 0.499 | 0.489 | 0.506 |
| 2048 | 1.199 | 0.890 | 1.079 | 1.080 | 1.090 |
| 4096 | 2.412 | 2.287 | 2.274 | 2.261 | 2.387 |
| 8192 | 5.293 | 5.252 | 4.899 | 5.056 | 5.454 |
| 16384 | 12.851 | 12.736 | 10.861 | 12.015 | 11.383 |
| 32768 | 32.406 | 25.132 | 27.080 | 29.772 | 27.109 |

11–27% faster than vDSP up to 2048. From 4096 every pass reads and writes 128 KB, four transforms' worth, the whole data cache.


#### Four step

A separate file, removed after these measurements: the N/2 points as R rows of C, R point FFTs down the columns, the twiddles W_M^(n2·k1), C point FFTs along the rows, each small FFT in scratch buffers of a few KB, the split reading the result transposed. The large buffers go through twice instead of six times. Correct at every size.

Its steps, four transforms at N = 4096, 32 rows of 64:

| | µs |
|---|---|
| column FFTs alone | 3.65 |
| column FFTs, twiddles, stored a row at a time | 7.02 |
| row FFTs | 3.87 |
| whole | 12.97, `run4` 9.5 |

The stores went 2 KB apart, a power of two, into the same few cache sets. At 16384 the column FFTs' reads, 4 KB apart, did too: 23.6 µs from the samples against 14.3 from a contiguous copy. Eight columns at a time, copied in and written out a row at a time, every access contiguous:

| N, a transform | `run4` | four step | four step, 8 columns at a time | vDSP FFT, 4 at once |
|---|---|---|---|---|
| 4096 | 2.36 | 3.17 | 3.37 | 2.45 |
| 8192 | 5.33 | 7.75 | 6.96 | 5.55 |
| 16384 | 12.96 | 23.98 | 15.82 | 11.79 |
| 32768 | 25.35 | 51.76 | 34.39 | 27.49 |

The small FFTs in the data cache take as long as `run4`'s passes through the next cache level, 7.5 µs at 4096, so the twiddle pass and the reordering are all extra.


#### One transform as four

[quad.odin](quad.odin): a real signal read four samples at a time is four signals interleaved, x[4i + l] in lane l, which `run4` takes as it is. Then one pass puts the four quarter length spectra together, X[k + j·N/4] = Σ_l W_N^(l·k)·W_4^(l·j)·X_l[k]: four k at a time, a 4×4 transpose turning the lanes from sub-signals into bins, the twiddles, a radix-4 butterfly whose four outputs are bins k, k + N/4, N/4 − k and N/2 − k (each X_l is a real signal's spectrum, X_l[N/4 − k] = conj(X_l[k])). The transpose and turning the two mirrored outputs around are the only reshuffles.

| N | `run` | `run_quad` | `run_quad_arrays` | vDSP FFT | vDSP DFT |
|---|---|---|---|---|---|
| 512 | 0.385 | 0.221 | 0.218 | 0.244 | 0.236 |
| 1024 | 0.552 | 0.527 | 0.523 | 0.506 | 0.499 |
| 2048 | 1.231 | 1.011 | 0.992 | 1.116 | 1.119 |
| 4096 | 2.457 | 2.400 | 2.362 | 2.360 | 2.345 |
| 8192 | 5.680 | 5.106 | 5.108 | 4.957 | 5.273 |
| 16384 | 13.250 | 12.667 | 12.629 | 11.251 | 12.536 |
| 32768 | 35.519 | 27.256 | 27.944 | 27.988 | 31.320 |

Faster than `run` at every size, about level with vDSP. The combining pass costs about two ordinary passes, 0.6 µs at 4096: for 16 bins a transpose of the real and the imaginary parts, 16 shuffles, three twiddle multiplies, the butterfly and two lane reversals, about 54 vector operations. The single transform's `ld2`, `st4` and reversals were the same kind of cost, mixing the four lanes takes one transpose's worth wherever it is done.

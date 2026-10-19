### Extreme FFT

A forward FFT of real samples in [Odin](https://odin-lang.org/), one file, no dependencies. N a power of two, 128 or more. Single precision, `#simd[4]f32`: NEON on ARM, SSE on x86, SIMD128 on wasm.

Written by artificial intelligence (Claude, by Anthropic) in October 2026, starting from the AI FFT in [fft-fun](https://github.com/dsego/fft-fun). [LOG.md](LOG.md) has each step and what it measured, the ones that didn't work too. The comments in [extreme_fft.odin](extreme_fft.odin) list what it does.


#### Use

```odin
import fft "extreme_fft"

plan := fft.create_plan(4096)
defer fft.destroy_plan(&plan)

// N/2 bins, bin 0 packs DC and Nyquist as (DC, Nyquist), like pffft_transform_ordered
spectrum := make([]complex64, 2048)
fft.run(&plan, samples, spectrum)

// Or into a real and an imaginary array like vDSP, DC in re[0] and Nyquist in im[0]
fft.run_arrays(&plan, samples, re, im)
```

Not scaled. A plan's work buffers are its own, one plan for each thread.

[lanes.odin](lanes.odin) does four transforms at once, one in each SIMD lane. The twiddles are the same for all four, so it needs none of the single transform's reshuffles. The four signals come interleaved, sample n of signal t at `samples[4n + t]`, the way four channel audio does, and the bins go out the same way, the four real parts of bin k at `spectrum[8k]` and the four imaginary parts at `spectrum[8k + 4]`:

```odin
plan := fft.create_lanes_plan(1024)
defer fft.destroy_lanes_plan(&plan)
fft.run_lanes(&plan, interleaved, spectrum) // 4·N floats in, 4·N floats out
```

[quad.odin](quad.odin) does one transform as four. A signal read four samples at a time is four signals interleaved, x[4i + l] in lane l, so `run_lanes` takes it as it is, and one more pass puts the four quarter length spectra together. The only reshuffle left is a 4×4 transpose in that pass. Faster than `run` at every size, N from 256, same outputs:

```odin
plan := fft.create_quad_plan(4096)
defer fft.destroy_quad_plan(&plan)
fft.run_quad(&plan, samples, spectrum)  // like run
fft.run_quad_arrays(&plan, samples, re, im)  // like run_arrays
```


#### Fused multiply-adds

Used on ARM64. On x86 they need FMA3, and without it LLVM calls the C library's `fmaf` for each lane, so they're off there unless asked for:

```
odin build . -define:EXTREME_FFT_FMA=true -microarch:native
```


#### Tests

Every way of computing a transform against a plain radix-2 FFT in f64, every power of two from 128 to 65536 (from 64 for four at once), and against the definition of the DFT alone: a direct sum in f64 up to 2048, an impulse, all ones and alternating ±1 (DC and Nyquist), tones on whole bins, Parseval's theorem and the shift theorem. The largest error is 3.5e-7 of the largest bin, the impulse, all ones and alternating ±1 come out exact. With and without fused multiply-adds:

```
odin test tests
odin test tests -define:EXTREME_FFT_FMA=false
```


#### Benchmark

Against Apple's vDSP on macOS, which takes the samples split into even and odd ones first (`vDSP_ctoz`, timed with it), one transform at a time and four in one call (`vDSP_fftm_zrip`). Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, microseconds a transform, the fastest of 5 runs of 20,000.

```
➜ odin run benchmark -o:speed -microarch:native
```

| N | `run` | `run_arrays` | `run_quad` | `run_quad_arrays` | `run_lanes`, a transform | vDSP FFT | vDSP DFT | vDSP FFT, 4 at once |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.385 | 0.338 | 0.221 | 0.218 | 0.174 | 0.244 | 0.236 | 0.256 |
| 1024 | 0.552 | 0.545 | 0.527 | 0.523 | 0.442 | 0.506 | 0.499 | 0.517 |
| 2048 | 1.231 | 1.199 | 1.011 | 0.992 | 0.910 | 1.116 | 1.119 | 1.119 |
| 4096 | 2.457 | 2.430 | 2.400 | 2.362 | 2.364 | 2.360 | 2.345 | 2.464 |
| 8192 | 5.680 | 5.443 | 5.106 | 5.108 | 5.393 | 4.957 | 5.273 | 5.659 |
| 16384 | 13.250 | 13.565 | 12.667 | 12.629 | 13.440 | 11.251 | 12.536 | 11.991 |
| 32768 | 35.519 | 40.204 | 27.256 | 27.944 | 25.952 | 27.988 | 31.320 | 28.060 |

One transform as four is about level with vDSP, ahead at 512, 2048 and 32768, 1–13% behind between. Four at once is faster than vDSP up to 2048. From 4096 its buffers, four transforms' worth, are larger than the M1's 128 KB data cache, and from 16384 a single transform's are too. strobe-tuner's own pffft build is 1–19% slower than `run` from 4096 to 65536.

Against the C libraries in [fft-fun](https://github.com/dsego/fft-fun), its benchmark with `run` and `run_quad` added (not in this repo). A 4096-point real FFT, microseconds, the fastest of 5 runs of 1,000 after a warm-up run, three runs of the benchmark. Same machine and Odin. FFTW is built with NEON. FFTW and Pocket FFT are double precision, the rest single. Both outputs match pffft's to 6.8e-08 of the largest bin.

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| **Extreme FFT, `run_quad`** | **2.335 µs** | **2.332 µs** | **2.332 µs** |
| **Extreme FFT, `run`** | **2.391 µs** | **2.391 µs** | **2.398 µs** |
| Pffft | 2.935 µs | 2.935 µs | 2.948 µs |
| FFTW3 (NEON) | 4.493 µs | 4.493 µs | 4.520 µs |
| Pocket FFT | 7.830 µs | 7.825 µs | 7.928 µs |
| Meow FFT | 10.362 µs | 10.282 µs | 10.409 µs |
| MUFFT | 10.645 µs | 10.556 µs | 10.618 µs |
| KissFFT real | 15.050 µs | 15.009 µs | 15.032 µs |

Tried and slower: splitting a transform into rows and columns that each fit the data cache (four step), the twiddles in tangent form, radix-8. [LOG.md](LOG.md) has them all.

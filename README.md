### Extreme FFT

A forward FFT of real samples in [Odin](https://odin-lang.org/), one file, no dependencies. N a power of two, 128 or more. Single precision, `#simd[4]f32`: NEON on ARM, SSE on x86, SIMD128 on wasm.

Written by artificial intelligence (Claude, by Anthropic) in October 2026, in [fft-fun](https://github.com/dsego/fft-fun) as `fft/fft_extreme.odin`, where `benchmark_stages/LOG.md` has each step that led to it and what it measured. The comments in [extreme_fft.odin](extreme_fft.odin) list what it does.


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

[four.odin](four.odin) does four transforms at once, one in each SIMD lane. The twiddles are the same for all four, so it needs none of the single transform's reshuffles. The four signals come interleaved, sample n of signal t at `samples[4n + t]`, the way four channel audio does, and the bins go out the same way, the four real parts of bin k at `spectrum[8k]` and the four imaginary parts at `spectrum[8k + 4]`:

```odin
plan := fft.create_plan4(1024)
defer fft.destroy_plan4(&plan)
fft.run4(&plan, interleaved, spectrum) // 4·N floats in, 4·N floats out
```


#### Fused multiply-adds

Used on ARM64. On x86 they need FMA3, and without it LLVM calls the C library's `fmaf` for each lane, so they're off there unless asked for:

```
odin build . -define:EXTREME_FFT_FMA=true -microarch:native
```


#### Tests

Against a plain radix-2 FFT in f64, every power of two from 128 to 65536 (from 64 for four at once), with and without fused multiply-adds:

```
odin test tests
odin test tests -define:EXTREME_FFT_FMA=false
```


#### Benchmark

Against Apple's vDSP on macOS, which takes the samples split into even and odd ones first (`vDSP_ctoz`, timed with it), one transform at a time and four in one call (`vDSP_fftm_zrip`). Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, microseconds a transform, the fastest of 5 runs of 20,000.

```
➜ odin run benchmark -o:speed -microarch:native
```

| N | extreme FFT | extreme, arrays | extreme, 4 lanes | vDSP FFT | vDSP DFT | vDSP FFT, 4 at once |
|---|---|---|---|---|---|---|
| 512 | 0.339 | 0.334 | 0.172 | 0.243 | 0.234 | 0.248 |
| 1024 | 0.538 | 0.528 | 0.436 | 0.499 | 0.489 | 0.506 |
| 2048 | 1.199 | 1.177 | 0.890 | 1.079 | 1.080 | 1.090 |
| 4096 | 2.412 | 2.370 | 2.287 | 2.274 | 2.261 | 2.387 |
| 8192 | 5.293 | 5.254 | 5.252 | 4.899 | 5.056 | 5.454 |
| 16384 | 12.851 | 12.949 | 12.736 | 10.861 | 12.015 | 11.383 |
| 32768 | 32.406 | 37.356 | 25.132 | 27.080 | 29.772 | 27.109 |

Four at once is 11–27% faster than vDSP up to 2048. From 4096 its buffers, four transforms' worth, are larger than the M1's 128 KB data cache, and from 16384 a single transform's are too. strobe-tuner's own pffft build is 1–19% slower than the extreme FFT from 4096 to 65536.

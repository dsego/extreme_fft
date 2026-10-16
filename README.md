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


#### Fused multiply-adds

Used on ARM64. On x86 they need FMA3, and without it LLVM calls the C library's `fmaf` for each lane, so they're off there unless asked for:

```
odin build . -define:EXTREME_FFT_FMA=true -microarch:native
```


#### Tests

Against a plain radix-2 FFT in f64, every power of two from 128 to 65536, with and without fused multiply-adds:

```
odin test tests
odin test tests -define:EXTREME_FFT_FMA=false
```


#### Benchmark

Against Apple's vDSP on macOS, which takes the samples split into even and odd ones first (`vDSP_ctoz`, timed with it). Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, microseconds, the fastest of 5 runs of 20,000.

```
➜ odin run benchmark -o:speed -microarch:native
```

| N | extreme FFT | extreme, arrays | vDSP FFT | vDSP DFT |
|---|---|---|---|---|
| 1024 | 0.565 | 0.527 | 0.500 | 0.490 |
| 4096 | 2.410 | 2.371 | 2.286 | 2.260 |
| 8192 | 5.322 | 5.316 | 4.914 | 5.089 |
| 16384 | 12.704 | 12.866 | 10.740 | 11.882 |
| 32768 | 32.414 | 37.354 | 26.935 | 29.734 |

pffft's ordered transform is a little slower than the extreme FFT up to 8192, see fft-fun. Past 8192 the work buffers no longer fit the M1's 128 KB data cache and vDSP pulls further ahead.

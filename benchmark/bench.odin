package benchmark

/*
    Written by Claude, an AI model made by Anthropic, October 2026.

    The extreme FFT, both outputs, against Apple's vDSP on macOS. vDSP's FFT and DFT take the
    samples split into even and odd ones first, vDSP_ctoz, timed with them. Microseconds per
    transform, the fastest of 5 runs.

        odin run benchmark -o:speed -microarch:native
*/

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:time"

import fft ".."

ITERATIONS :: 20_000
RUNS :: 5

Args :: struct {
    plan:     ^fft.Plan,
    samples:  []f32,
    spectrum: []complex64,
    re, im:   []f32,
    even, odd: []f32,
    setup:    rawptr,
    log2n:    uint,
}

main :: proc() {
    when ODIN_OS != .Darwin do fmt.println("vDSP is macOS only, the extreme FFT alone\n")

    for size in ([]int{1024, 4096, 8192, 16384, 32768}) {
        samples := make_floats(size)
        defer delete(samples)
        for &sample, i in samples {
            t := f32(i)
            sample = 0.6 * math.sin(0.031 * t) + 0.3 * math.sin(0.47 * t + 1) + 0.1 * math.cos(2.9 * t)
        }

        plan := fft.create_plan(size)
        defer fft.destroy_plan(&plan)
        spectrum := make([]complex64, size / 2)
        defer delete(spectrum)
        re, im := make_floats(size / 2), make_floats(size / 2)
        defer delete(re)
        defer delete(im)

        fmt.printf("N = %v\n", size)
        row("extreme FFT", best_of(proc(a: ^Args) { fft.run(a.plan, a.samples, a.spectrum) }, &Args{plan = &plan, samples = samples, spectrum = spectrum}))
        row("extreme, arrays", best_of(proc(a: ^Args) { fft.run_arrays(a.plan, a.samples, a.re, a.im) }, &Args{plan = &plan, samples = samples, re = re, im = im}))

        when ODIN_OS == .Darwin {
            log2n := uint(0)
            for 1 << log2n < size do log2n += 1
            vdsp_fft := vDSP_create_fftsetup(log2n, FFT_RADIX_2)
            defer vDSP_destroy_fftsetup(vdsp_fft)
            vdsp_dft := vDSP_DFT_zrop_CreateSetup(nil, uint(size), DFT_FORWARD)
            defer vDSP_DFT_DestroySetup(vdsp_dft)
            even, odd := make_floats(size / 2), make_floats(size / 2)
            defer delete(even)
            defer delete(odd)

            row("vDSP FFT", best_of(proc(a: ^Args) {
                split := DSPSplitComplex{raw_data(a.even), raw_data(a.odd)}
                vDSP_ctoz(raw_data(a.samples), 2, &split, 1, uint(len(a.even)))
                vDSP_fft_zrip(a.setup, &split, 1, a.log2n, FFT_FORWARD)
            }, &Args{samples = samples, even = even, odd = odd, setup = vdsp_fft, log2n = log2n}))
            row("vDSP DFT", best_of(proc(a: ^Args) {
                split := DSPSplitComplex{raw_data(a.even), raw_data(a.odd)}
                vDSP_ctoz(raw_data(a.samples), 2, &split, 1, uint(len(a.even)))
                vDSP_DFT_Execute(a.setup, raw_data(a.even), raw_data(a.odd), raw_data(a.re), raw_data(a.im))
            }, &Args{samples = samples, even = even, odd = odd, re = re, im = im, setup = vdsp_dft}))
        }
        fmt.println()
    }
}

row :: proc(name: string, µs: f64) {
    fmt.printf("  %-18s % 8.3f µs\n", name, µs)
}

// Microseconds per call, the fastest of RUNS runs of ITERATIONS calls after a warm-up
best_of :: proc(run: proc(a: ^Args), args: ^Args) -> f64 {
    for _ in 0 ..< 1000 do run(args)
    best := max(f64)
    for _ in 0 ..< RUNS {
        stopwatch: time.Stopwatch
        time.stopwatch_start(&stopwatch)
        for _ in 0 ..< ITERATIONS do run(args)
        time.stopwatch_stop(&stopwatch)
        best = min(best, time.duration_microseconds(time.stopwatch_duration(stopwatch)) / ITERATIONS)
    }
    return best
}

make_floats :: proc(count: int) -> []f32 {
    slice, _ := runtime.make_aligned([]f32, count, 16)
    return slice
}

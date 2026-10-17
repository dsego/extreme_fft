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
    plan4:    ^fft.Plan4,
    quad:     ^fft.Quad_Plan,
    samples:  []f32,
    spectrum: []complex64,
    re, im:   []f32,
    even, odd: []f32,
    setup:    rawptr,
    log2n:    uint,
}

main :: proc() {
    when ODIN_OS != .Darwin do fmt.println("vDSP is macOS only, the extreme FFT alone\n")

    for size in ([]int{512, 1024, 2048, 4096, 8192, 16384, 32768}) {
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

        // Four signals for the four lane FFT, interleaved, and for vDSP's batch one after another
        plan4 := fft.create_plan4(size)
        defer fft.destroy_plan4(&plan4)
        interleaved, four_spectra := make_floats(4 * size), make_floats(4 * size)
        defer delete(interleaved)
        defer delete(four_spectra)
        for sample, n in samples do for lane in 0 ..< 4 do interleaved[4 * n + lane] = sample * f32(lane + 1)

        fmt.printf("N = %v, microseconds a transform\n", size)
        row("extreme FFT", best_of(proc(a: ^Args) { fft.run(a.plan, a.samples, a.spectrum) }, &Args{plan = &plan, samples = samples, spectrum = spectrum}))
        row("extreme, arrays", best_of(proc(a: ^Args) { fft.run_arrays(a.plan, a.samples, a.re, a.im) }, &Args{plan = &plan, samples = samples, re = re, im = im}))
        row("extreme, 4 lanes", best_of(proc(a: ^Args) { fft.run4(a.plan4, a.samples, a.re) }, &Args{plan4 = &plan4, samples = interleaved, re = four_spectra}) / 4)
        quad := fft.create_quad_plan(size)
        defer fft.destroy_quad_plan(&quad)
        row("quad", best_of(proc(a: ^Args) { fft.run_quad(a.quad, a.samples, a.spectrum) }, &Args{quad = &quad, samples = samples, spectrum = spectrum}))
        row("quad, arrays", best_of(proc(a: ^Args) { fft.run_quad_arrays(a.quad, a.samples, a.re, a.im) }, &Args{quad = &quad, samples = samples, re = re, im = im}))

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

            four_signals := make_floats(4 * size)
            defer delete(four_signals)
            for sample, n in samples do for lane in 0 ..< 4 do four_signals[lane * size + n] = sample * f32(lane + 1)
            four_even, four_odd := make_floats(2 * size), make_floats(2 * size)
            defer delete(four_even)
            defer delete(four_odd)
            row("vDSP FFT, batch of 4", best_of(proc(a: ^Args) {
                half := len(a.even) / 4
                split := DSPSplitComplex{raw_data(a.even), raw_data(a.odd)}
                for lane in 0 ..< 4 {
                    lane_split := DSPSplitComplex{raw_data(a.even[lane * half:]), raw_data(a.odd[lane * half:])}
                    vDSP_ctoz(raw_data(a.samples[2 * lane * half:]), 2, &lane_split, 1, uint(half))
                }
                vDSP_fftm_zrip(a.setup, &split, 1, half, a.log2n, 4, FFT_FORWARD)
            }, &Args{samples = four_signals, even = four_even, odd = four_odd, setup = vdsp_fft, log2n = log2n}) / 4)
        }
        fmt.println()
    }
}

row :: proc(name: string, µs: f64) {
    fmt.printf("  %-22s % 8.3f µs\n", name, µs)
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

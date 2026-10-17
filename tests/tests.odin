package tests

/*
    Written by Claude, an AI model made by Anthropic, October 2026.

    The extreme FFT against a plain radix-2 FFT in f64, every power of two from 128 to 65536.

        odin test tests
*/

import "core:math"
import "core:math/rand"
import "core:testing"

import fft ".."

// The largest error, relative to the largest bin, an f32 transform should stay under
TOLERANCE :: 1e-6

@(test)
test_run :: proc(t: ^testing.T) {
    for size := 128; size <= 65536; size *= 2 {
        samples, expected := signal(size)
        defer delete(samples)
        defer delete(expected)

        plan := fft.create_plan(size)
        defer fft.destroy_plan(&plan)
        spectrum := make([]complex64, size / 2)
        defer delete(spectrum)
        fft.run(&plan, samples, spectrum)

        // Bin 0 packs DC and Nyquist
        got := make([]complex128, size / 2 + 1)
        defer delete(got)
        got[0] = complex(f64(real(spectrum[0])), 0)
        got[size / 2] = complex(f64(imag(spectrum[0])), 0)
        for k in 1 ..< size / 2 do got[k] = complex128(spectrum[k])

        error := relative_error(got, expected)
        testing.expectf(t, error < TOLERANCE, "run, N = %v: %.2e of the largest bin", size, error)
    }
}

@(test)
test_run_arrays :: proc(t: ^testing.T) {
    for size := 128; size <= 65536; size *= 2 {
        samples, expected := signal(size)
        defer delete(samples)
        defer delete(expected)

        plan := fft.create_plan(size)
        defer fft.destroy_plan(&plan)
        re, im := make([]f32, size / 2), make([]f32, size / 2)
        defer delete(re)
        defer delete(im)
        fft.run_arrays(&plan, samples, re, im)

        // DC in re[0], Nyquist in im[0]
        got := make([]complex128, size / 2 + 1)
        defer delete(got)
        got[0] = complex(f64(re[0]), 0)
        got[size / 2] = complex(f64(im[0]), 0)
        for k in 1 ..< size / 2 do got[k] = complex(f64(re[k]), f64(im[k]))

        error := relative_error(got, expected)
        testing.expectf(t, error < TOLERANCE, "run_arrays, N = %v: %.2e of the largest bin", size, error)
    }
}

// Four signals at once, interleaved, each lane against its own reference
@(test)
test_run4 :: proc(t: ^testing.T) {
    for size := 64; size <= 65536; size *= 2 {
        samples := make([]f32, 4 * size)
        defer delete(samples)
        expected: [4][]complex128
        for &lane_expected, lane in expected {
            lane_samples: []f32
            lane_samples, lane_expected = signal(size, u64(lane))
            for sample, n in lane_samples do samples[4 * n + lane] = sample
            delete(lane_samples)
        }
        defer for e in expected do delete(e)

        plan := fft.create_plan4(size)
        defer fft.destroy_plan4(&plan)
        spectrum := make([]f32, 4 * size)
        defer delete(spectrum)
        fft.run4(&plan, samples, spectrum)

        got := make([]complex128, size / 2 + 1)
        defer delete(got)
        for lane in 0 ..< 4 {
            // Bin k's real parts at 8k, imaginary parts at 8k + 4. DC and Nyquist in bin 0.
            got[0] = complex(f64(spectrum[lane]), 0)
            got[size / 2] = complex(f64(spectrum[4 + lane]), 0)
            for k in 1 ..< size / 2 do got[k] = complex(f64(spectrum[8 * k + lane]), f64(spectrum[8 * k + 4 + lane]))
            error := relative_error(got, expected[lane])
            testing.expectf(t, error < TOLERANCE, "run4, N = %v, lane %v: %.2e of the largest bin", size, lane, error)
        }
    }
}

// A plan used again and again gives the same bins, its buffers carry nothing over
@(test)
test_repeat :: proc(t: ^testing.T) {
    SIZE :: 4096
    samples, expected := signal(SIZE)
    defer delete(samples)
    defer delete(expected)
    plan := fft.create_plan(SIZE)
    defer fft.destroy_plan(&plan)

    first, again := make([]complex64, SIZE / 2), make([]complex64, SIZE / 2)
    defer delete(first)
    defer delete(again)
    fft.run(&plan, samples, first)
    for _ in 0 ..< 3 do fft.run(&plan, samples, again)
    for bin, k in first do testing.expectf(t, bin == again[k], "bin %v: %v, then %v", k, bin, again[k])
}

// Noise and a few tones, and its spectrum from a plain f64 FFT: bins 0 to N/2. A different
// variant gives different noise.
@(private = "file")
signal :: proc(size: int, variant: u64 = 0) -> (samples: []f32, spectrum: []complex128) {
    generator := rand.create(u64(size) + 1000 * variant)
    context.random_generator = rand.default_random_generator(&generator)

    samples = make([]f32, size)
    for &sample, i in samples {
        t := f64(i)
        tones := 0.6 * math.sin(0.031 * t) + 0.3 * math.sin(0.47 * t + 1) + 0.1 * math.cos(2.9 * t)
        sample = f32(tones + 0.2 * (rand.float64() - 0.5))
    }

    full := make([]complex128, size)
    defer delete(full)
    for sample, i in samples do full[i] = complex(f64(sample), 0)
    reference_fft(full)
    spectrum = make([]complex128, size / 2 + 1)
    copy(spectrum, full[:size / 2 + 1])
    return
}

// In place, iterative radix-2 with the twiddles worked out in f64 for each one
@(private = "file")
reference_fft :: proc(x: []complex128) {
    n := len(x)
    j := 0
    for i in 1 ..< n {
        bit := n >> 1
        for j & bit != 0 {
            j ~= bit
            bit >>= 1
        }
        j |= bit
        if i < j do x[i], x[j] = x[j], x[i]
    }
    for length := 2; length <= n; length *= 2 {
        for start := 0; start < n; start += length {
            for k in 0 ..< length / 2 {
                angle := -math.TAU * f64(k) / f64(length)
                w := complex(math.cos(angle), math.sin(angle))
                a, b := x[start + k], w * x[start + k + length / 2]
                x[start + k], x[start + k + length / 2] = a + b, a - b
            }
        }
    }
}

// The largest difference relative to the largest bin
@(private = "file")
relative_error :: proc(got, expected: []complex128) -> f64 {
    worst, peak: f64
    for value, k in expected {
        peak = max(peak, abs(value))
        worst = max(worst, abs(got[k] - value))
    }
    return worst / peak
}

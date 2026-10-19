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
test_run_lanes :: proc(t: ^testing.T) {
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

        plan := fft.create_lanes_plan(size)
        defer fft.destroy_lanes_plan(&plan)
        spectrum := make([]f32, 4 * size)
        defer delete(spectrum)
        fft.run_lanes(&plan, samples, spectrum)

        got := make([]complex128, size / 2 + 1)
        defer delete(got)
        for lane in 0 ..< 4 {
            // Bin k's real parts at 8k, imaginary parts at 8k + 4. DC and Nyquist in bin 0.
            got[0] = complex(f64(spectrum[lane]), 0)
            got[size / 2] = complex(f64(spectrum[4 + lane]), 0)
            for k in 1 ..< size / 2 do got[k] = complex(f64(spectrum[8 * k + lane]), f64(spectrum[8 * k + 4 + lane]))
            error := relative_error(got, expected[lane])
            testing.expectf(t, error < TOLERANCE, "run_lanes, N = %v, lane %v: %.2e of the largest bin", size, lane, error)
        }
    }
}

// One transform done as four, both outputs
@(test)
test_run_quad :: proc(t: ^testing.T) {
    for size := 256; size <= 65536; size *= 2 {
        samples, expected := signal(size)
        defer delete(samples)
        defer delete(expected)
        plan := fft.create_quad_plan(size)
        defer fft.destroy_quad_plan(&plan)

        spectrum := make([]complex64, size / 2)
        defer delete(spectrum)
        fft.run_quad(&plan, samples, spectrum)
        got := make([]complex128, size / 2 + 1)
        defer delete(got)
        got[0] = complex(f64(real(spectrum[0])), 0)
        got[size / 2] = complex(f64(imag(spectrum[0])), 0)
        for k in 1 ..< size / 2 do got[k] = complex128(spectrum[k])
        error := relative_error(got, expected)
        testing.expectf(t, error < TOLERANCE, "run_quad, N = %v: %.2e of the largest bin", size, error)

        re, im := make([]f32, size / 2), make([]f32, size / 2)
        defer delete(re)
        defer delete(im)
        fft.run_quad_arrays(&plan, samples, re, im)
        got[0] = complex(f64(re[0]), 0)
        got[size / 2] = complex(f64(im[0]), 0)
        for k in 1 ..< size / 2 do got[k] = complex(f64(re[k]), f64(im[k]))
        error = relative_error(got, expected)
        testing.expectf(t, error < TOLERANCE, "run_quad_arrays, N = %v: %.2e of the largest bin", size, error)
    }
}

/*
    Checks against the definition of the DFT alone, X[k] = Σ x[n]·e^(-2πi·kn/N), no FFT code in them: a
    direct sum, signals whose spectrum is known exactly, Parseval's theorem and the shift theorem. Each one
    runs every way the package has to compute a transform.
*/

// The largest sizes for the direct sum, O(N²), and for the exact spectra and the shift
DIRECT_MAX :: 2048
EXACT_MAX :: 16384

// The direct sum in f64, bins 0 to N/2, the twiddles exact from a table of N
@(test)
test_direct_dft :: proc(t: ^testing.T) {
    for size := 128; size <= DIRECT_MAX; size *= 2 {
        samples, spectrum := signal(size, 7)
        delete(spectrum)
        defer delete(samples)

        cos_table, sin_table := make([]f64, size), make([]f64, size)
        defer delete(cos_table)
        defer delete(sin_table)
        for n in 0 ..< size {
            cos_table[n] = math.cos(math.TAU * f64(n) / f64(size))
            sin_table[n] = math.sin(math.TAU * f64(n) / f64(size))
        }
        expected := make([]complex128, size / 2 + 1)
        defer delete(expected)
        for k in 0 ..= size / 2 {
            re, im: f64
            for x, n in samples {
                i := (k * n) % size
                re += f64(x) * cos_table[i]
                im -= f64(x) * sin_table[i]
            }
            expected[k] = complex(re, im)
        }

        results := all_transforms(samples)
        defer delete_transforms(results)
        for result in results {
            error := relative_error(result.bins, expected)
            testing.expectf(t, error < TOLERANCE, "%v against the direct DFT, N = %v: %.2e of the largest bin", result.name, size, error)
        }
    }
}

// An impulse at 0 is 1 in every bin, at n0 the phase e^(-2πi·k·n0/N). All ones is N in the DC bin and
// nothing else, alternating ±1 N in the Nyquist bin: those two are the bins packed together in bin 0.
@(test)
test_exact_spectra :: proc(t: ^testing.T) {
    for size := 128; size <= EXACT_MAX; size *= 2 {
        half := size / 2
        samples := make([]f32, size)
        defer delete(samples)
        expected := make([]complex128, half + 1)
        defer delete(expected)

        cases := [?]string{"impulse at 0", "impulse at 7", "all ones", "alternating ±1"}
        for name, which in cases {
            for &x in samples do x = 0
            for &e in expected do e = 0
            switch which {
            case 0, 1:
                n0 := 0 if which == 0 else 7
                samples[n0] = 1
                for k in 0 ..= half {
                    angle := -math.TAU * f64(k * n0 % size) / f64(size)
                    expected[k] = complex(math.cos(angle), math.sin(angle))
                }
            case 2:
                for &x in samples do x = 1
                expected[0] = complex(f64(size), 0)
            case 3:
                for &x, n in samples do x = 1 if n % 2 == 0 else -1
                expected[half] = complex(f64(size), 0)
            }

            results := all_transforms(samples)
            defer delete_transforms(results)
            for result in results {
                error := relative_error(result.bins, expected)
                testing.expectf(t, error < TOLERANCE, "%v, %v, N = %v: %.2e of the largest bin", result.name, name, size, error)
            }
        }
    }
}

// cos(2π·k1·n/N) + ½·sin(2π·k2·n/N) with k1 and k2 whole bins: N/2 in bin k1, -j·N/4 in bin k2, nothing
// anywhere else
@(test)
test_tones_on_bins :: proc(t: ^testing.T) {
    for size := 128; size <= EXACT_MAX; size *= 2 {
        k1, k2 := size / 8 + 3, size / 4 - 5
        samples := make([]f32, size)
        defer delete(samples)
        for &x, n in samples {
            x = f32(math.cos(math.TAU * f64(k1 * n % size) / f64(size)) + 0.5 * math.sin(math.TAU * f64(k2 * n % size) / f64(size)))
        }
        expected := make([]complex128, size / 2 + 1)
        defer delete(expected)
        expected[k1] = complex(f64(size) / 2, 0)
        expected[k2] = complex(0, -f64(size) / 4)

        results := all_transforms(samples)
        defer delete_transforms(results)
        for result in results {
            error := relative_error(result.bins, expected)
            testing.expectf(t, error < TOLERANCE, "%v, tones on bins %v and %v, N = %v: %.2e of the largest bin", result.name, k1, k2, size, error)
        }
    }
}

// Parseval: Σ x[n]² = (|X[0]|² + |X[N/2]|² + 2·Σ |X[k]|² for 0 < k < N/2) / N
@(test)
test_parseval :: proc(t: ^testing.T) {
    for size := 128; size <= 65536; size *= 2 {
        samples, spectrum := signal(size, 11)
        delete(spectrum)
        defer delete(samples)
        energy: f64
        for x in samples do energy += f64(x) * f64(x)

        results := all_transforms(samples)
        defer delete_transforms(results)
        for result in results {
            bins := result.bins
            half := size / 2
            sum := abs(bins[0]) * abs(bins[0]) + abs(bins[half]) * abs(bins[half])
            for k in 1 ..< half do sum += 2 * abs(bins[k]) * abs(bins[k])
            difference := abs(sum / f64(size) - energy) / energy
            testing.expectf(t, difference < TOLERANCE, "%v, Parseval, N = %v: energies differ by %.2e", result.name, size, difference)
        }
    }
}

// The samples shifted around by m, x[(n - m) mod N], have the same bins times e^(-2πi·k·m/N)
@(test)
test_shift :: proc(t: ^testing.T) {
    M :: 5
    for size := 128; size <= EXACT_MAX; size *= 2 {
        samples, spectrum := signal(size, 13)
        delete(spectrum)
        defer delete(samples)
        shifted := make([]f32, size)
        defer delete(shifted)
        for n in 0 ..< size do shifted[n] = samples[(n - M + size) % size]

        results := all_transforms(samples)
        defer delete_transforms(results)
        shifted_results := all_transforms(shifted)
        defer delete_transforms(shifted_results)

        expected := make([]complex128, size / 2 + 1)
        defer delete(expected)
        for result, index in results {
            for k in 0 ..= size / 2 {
                angle := -math.TAU * f64(k * M % size) / f64(size)
                expected[k] = result.bins[k] * complex(math.cos(angle), math.sin(angle))
            }
            error := relative_error(shifted_results[index].bins, expected)
            testing.expectf(t, error < TOLERANCE, "%v, shifted by %v, N = %v: %.2e of the largest bin", result.name, M, size, error)
        }
    }
}

// One transform's bins 0 to N/2, and which way computed them
@(private = "file")
Transform :: struct {
    name: string,
    bins: []complex128,
}

// The samples through every way the package has: run and run_arrays, run_quad and run_quad_arrays from
// 256, and run_lanes with the samples in all four lanes, lane l times l + 1 and divided back, so a lane
// picking up another's bins shows
@(private = "file")
all_transforms :: proc(samples: []f32) -> (results: [dynamic]Transform) {
    size := len(samples)
    half := size / 2

    from_packed :: proc(spectrum: []complex64) -> []complex128 {
        half := len(spectrum)
        bins := make([]complex128, half + 1)
        bins[0] = complex(f64(real(spectrum[0])), 0)
        bins[half] = complex(f64(imag(spectrum[0])), 0)
        for k in 1 ..< half do bins[k] = complex128(spectrum[k])
        return bins
    }
    from_arrays :: proc(re, im: []f32) -> []complex128 {
        half := len(re)
        bins := make([]complex128, half + 1)
        bins[0] = complex(f64(re[0]), 0)
        bins[half] = complex(f64(im[0]), 0)
        for k in 1 ..< half do bins[k] = complex(f64(re[k]), f64(im[k]))
        return bins
    }

    spectrum := make([]complex64, half)
    defer delete(spectrum)
    re, im := make([]f32, half), make([]f32, half)
    defer delete(re)
    defer delete(im)

    plan := fft.create_plan(size)
    defer fft.destroy_plan(&plan)
    fft.run(&plan, samples, spectrum)
    append(&results, Transform{"run", from_packed(spectrum)})
    fft.run_arrays(&plan, samples, re, im)
    append(&results, Transform{"run_arrays", from_arrays(re, im)})

    if size >= 256 {
        quad := fft.create_quad_plan(size)
        defer fft.destroy_quad_plan(&quad)
        fft.run_quad(&quad, samples, spectrum)
        append(&results, Transform{"run_quad", from_packed(spectrum)})
        fft.run_quad_arrays(&quad, samples, re, im)
        append(&results, Transform{"run_quad_arrays", from_arrays(re, im)})
    }

    interleaved, four := make([]f32, 4 * size), make([]f32, 4 * size)
    defer delete(interleaved)
    defer delete(four)
    for x, n in samples {
        for lane in 0 ..< 4 do interleaved[4 * n + lane] = x * f32(lane + 1)
    }
    lanes_plan := fft.create_lanes_plan(size)
    defer fft.destroy_lanes_plan(&lanes_plan)
    fft.run_lanes(&lanes_plan, interleaved, four)
    lanes := [4]string{"run_lanes lane 0", "run_lanes lane 1", "run_lanes lane 2", "run_lanes lane 3"}
    for lane in 0 ..< 4 {
        scale := f64(lane + 1)
        bins := make([]complex128, half + 1)
        bins[0] = complex(f64(four[lane]) / scale, 0)
        bins[half] = complex(f64(four[4 + lane]) / scale, 0)
        for k in 1 ..< half do bins[k] = complex(f64(four[8 * k + lane]), f64(four[8 * k + 4 + lane])) / complex(scale, 0)
        append(&results, Transform{lanes[lane], bins})
    }
    return
}

@(private = "file")
delete_transforms :: proc(results: [dynamic]Transform) {
    for result in results do delete(result.bins)
    delete(results)
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

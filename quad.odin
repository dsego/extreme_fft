package extreme_fft

import "core:math"
import "core:math/bits"
import "core:simd"


/*
    ============================================================================================
    WRITTEN BY ARTIFICIAL INTELLIGENCE: this file was written by Claude, an AI model made by
    Anthropic, in October 2026.

    One transform as four. A real signal read four samples at a time is four signals
    interleaved, x_l[i] = x[4i + l] in lane l, exactly what lanes.odin's run_lanes takes.
    run_lanes needs none of a single transform's reshuffles: no even and odd split, no
    transpose, no turning lanes around. Then one pass puts the four quarter length spectra
    together:

        X[k + j·N/4] = Σ_l W_N^(l·k) · W_4^(l·j) · X_l[k]

    For each k a radix-4 butterfly across the lanes, after the twiddles W_N^(l·k). Its four
    outputs are four bins, because each X_l is the spectrum of a real signal,
    X_l[N/4 - k] = conj(X_l[k]):

        X[k] = D0,  X[k + N/4] = D1,  X[N/4 - k] = conj(D3),  X[N/2 - k] = conj(D2)

    Four k at a time, the lanes turned from sub-signals into bins with one 4×4 transpose, the
    only reshuffle left besides turning the two mirrored outputs around.

    run_quad lays the bins out like run, bin 0 packing DC and Nyquist, run_quad_arrays like
    run_arrays. Forward only, not scaled, N a power of two, 256 or more.
    ============================================================================================
*/


// A plan for one transform done as four. Not for two threads at once.
Quad_Plan :: struct {
    size:       int, // N
    quarter:    int, // L = N/4, the length of each sub-signal
    sub:        Lanes_Plan, // the four sub-signals' FFTs
    sub_bins:   []f32, // their L/2 bins, 8 floats a bin, as run_lanes leaves them
    twiddle_re: []f32, // W_N^(l·k) for l = 1, 2, 3 and k < L/2, at (l - 1)·L/2 + k
    twiddle_im: []f32,
}

create_quad_plan :: proc(size: int, allocator := context.allocator) -> (plan: Quad_Plan) {
    assert(bits.is_power_of_two(size) && size >= 256)
    context.allocator = allocator

    plan.size = size
    plan.quarter = size / 4
    plan.sub = create_lanes_plan(plan.quarter)
    plan.sub_bins = make_floats(4 * plan.quarter)

    half_quarter := plan.quarter / 2
    plan.twiddle_re = make_floats(3 * half_quarter)
    plan.twiddle_im = make_floats(3 * half_quarter)
    for l in 1 ..= 3 {
        for k in 0 ..< half_quarter {
            angle := -math.TAU * f64(l * k) / f64(size)
            plan.twiddle_re[(l - 1) * half_quarter + k] = f32(math.cos(angle))
            plan.twiddle_im[(l - 1) * half_quarter + k] = f32(math.sin(angle))
        }
    }
    return
}

destroy_quad_plan :: proc(plan: ^Quad_Plan, allocator := context.allocator) {
    context.allocator = allocator
    destroy_lanes_plan(&plan.sub)
    delete(plan.sub_bins)
    delete(plan.twiddle_re)
    delete(plan.twiddle_im)
}

// N real samples to N/2 bins, bin 0 is (DC, Nyquist)
run_quad :: proc(plan: ^Quad_Plan, samples: []f32, spectrum: []complex64) {
    assert(len(samples) == plan.size && len(spectrum) == plan.size / 2)
    run_lanes(&plan.sub, samples, plan.sub_bins)
    out := cast([^]f32)raw_data(spectrum)
    combine(plan, out, nil, false)
}

// N real samples to N/2 bins in two arrays, DC in re[0] and Nyquist in im[0]
run_quad_arrays :: proc(plan: ^Quad_Plan, samples: []f32, re, im: []f32) {
    assert(len(samples) == plan.size && len(re) == plan.size / 2 && len(im) == plan.size / 2)
    run_lanes(&plan.sub, samples, plan.sub_bins)
    combine(plan, raw_data(re), raw_data(im), true)
}

/*
    The four sub-spectra put together. Four k at a time from 4 to L/2 - 4, the block at k = 0 and
    the single k = L/2 apart, where the sub-spectra's packed DC and Nyquist come in and the
    full spectrum's go out. With ARRAYS the bins go to out (real parts) and out_im, without into
    out interleaved.
*/
@(private = "file")
combine :: proc(plan: ^Quad_Plan, out, out_im: [^]f32, $ARRAYS: bool) #no_bounds_check {
    L := plan.quarter
    h := L / 2
    bins := raw_data(plan.sub_bins)
    tw_re, tw_im := raw_data(plan.twiddle_re), raw_data(plan.twiddle_im)

    for k := 4; k < h; k += 4 {
        // Bins k to k+3 of the four sub-signals, the lanes the sub-signals, turned into the four
        // sub-signals' bins k to k+3, the lanes k
        r0, r1, r2, r3 := load(bins[8 * k:]), load(bins[8 * k + 8:]), load(bins[8 * k + 16:]), load(bins[8 * k + 24:])
        i0, i1, i2, i3 := load(bins[8 * k + 4:]), load(bins[8 * k + 12:]), load(bins[8 * k + 20:]), load(bins[8 * k + 28:])
        a_re, b_re, c_re, d_re := transpose4(r0, r1, r2, r3)
        a_im, b_im, c_im, d_im := transpose4(i0, i1, i2, i3)

        d0_re, d0_im, d1_re, d1_im, d2_re, d2_im, d3_re, d3_im := twiddled_butterfly(
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, tw_re, tw_im, h, k,
        )
        store_bins(out, out_im, k, d0_re, d0_im, ARRAYS)
        store_bins(out, out_im, k + L, d1_re, d1_im, ARRAYS)
        store_bins_mirrored(out, out_im, L - k - 3, d3_re, -d3_im, ARRAYS)
        store_bins_mirrored(out, out_im, 2 * L - k - 3, d2_re, -d2_im, ARRAYS)
    }

    // k = 0 to 3: X_l[0] is the sub-signal's DC, real, its imaginary slot holds the Nyquist bin
    {
        r0, r1, r2, r3 := load(bins[0:]), load(bins[8:]), load(bins[16:]), load(bins[24:])
        i1, i2, i3 := load(bins[12:]), load(bins[20:]), load(bins[28:])
        a_re, b_re, c_re, d_re := transpose4(r0, r1, r2, r3)
        a_im, b_im, c_im, d_im := transpose4(F32x4{}, i1, i2, i3)

        d0_re, d0_im, d1_re, d1_im, d2_re, d2_im, d3_re, d3_im := twiddled_butterfly(
            a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im, tw_re, tw_im, h, 0,
        )
        x0_re, x0_im := simd.to_array(d0_re), simd.to_array(d0_im)
        x2_re, x2_im := simd.to_array(d2_re), simd.to_array(d2_im)
        x3_re, x3_im := simd.to_array(d3_re), simd.to_array(d3_im)
        x1_re, x1_im := simd.to_array(d1_re), simd.to_array(d1_im)

        // Lane 0 is k = 0: D0 is DC, D2 the Nyquist bin, both real; D1 and conj(D3) both X[L]
        set_bin(out, out_im, 0, x0_re[0], x2_re[0], ARRAYS)
        for lane in 1 ..< 4 {
            set_bin(out, out_im, lane, x0_re[lane], x0_im[lane], ARRAYS)
            set_bin(out, out_im, L - lane, x3_re[lane], -x3_im[lane], ARRAYS)
            set_bin(out, out_im, 2 * L - lane, x2_re[lane], -x2_im[lane], ARRAYS)
        }
        for lane in 0 ..< 4 do set_bin(out, out_im, L + lane, x1_re[lane], x1_im[lane], ARRAYS)
    }

    // k = L/2: X_l[L/2] is the sub-signal's Nyquist bin, real, and the twiddles are W_8^l, that
    // is 1, (1 - j)/√2, -j and (-1 - j)/√2. The four outputs are two bins, X[L/2] = D0 and
    // X[3L/2] = D1 = (Y0 - Y2) - j·(Y1 - Y3).
    {
        R :: f32(math.SQRT_TWO / 2)
        J :: complex64(1i)
        n := simd.to_array(load(bins[4:]))
        y0 := complex(n[0], 0)
        y1 := complex(R * n[1], -R * n[1])
        y2 := complex(0, -n[2])
        y3 := complex(-R * n[3], -R * n[3])
        d0 := (y0 + y2) + (y1 + y3)
        d1 := (y0 - y2) - J * (y1 - y3)
        set_bin(out, out_im, h, real(d0), imag(d0), ARRAYS)
        set_bin(out, out_im, L + h, real(d1), imag(d1), ARRAYS)
    }
}

// The butterfly across the four sub-signals for bins k to k+3, after the twiddles W_N^(l·k) on
// sub-signals 1, 2 and 3
@(private = "file")
twiddled_butterfly :: #force_inline proc "contextless" (
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im: F32x4,
    tw_re, tw_im: [^]f32,
    h, k: int,
) -> (
    d0_re, d0_im, d1_re, d1_im, d2_re, d2_im, d3_re, d3_im: F32x4,
) {
    wb_re, wb_im := cmul(b_re, b_im, load(tw_re[k:]), load(tw_im[k:]))
    wc_re, wc_im := cmul(c_re, c_im, load(tw_re[h + k:]), load(tw_im[h + k:]))
    wd_re, wd_im := cmul(d_re, d_im, load(tw_re[2 * h + k:]), load(tw_im[2 * h + k:]))
    return butterfly4(a_re, a_im, wb_re, wb_im, wc_re, wc_im, wd_re, wd_im)
}

// Rows r0..r3 as columns
@(private = "file")
transpose4 :: #force_inline proc "contextless" (r0, r1, r2, r3: F32x4) -> (c0, c1, c2, c3: F32x4) {
    t0 := simd.shuffle(r0, r1, 0, 4, 1, 5)
    t1 := simd.shuffle(r0, r1, 2, 6, 3, 7)
    t2 := simd.shuffle(r2, r3, 0, 4, 1, 5)
    t3 := simd.shuffle(r2, r3, 2, 6, 3, 7)
    return simd.shuffle(t0, t2, 0, 1, 4, 5), simd.shuffle(t0, t2, 2, 3, 6, 7), simd.shuffle(t1, t3, 0, 1, 4, 5), simd.shuffle(t1, t3, 2, 3, 6, 7)
}

// Bins first to first+3
@(private = "file")
store_bins :: #force_inline proc "contextless" (out, out_im: [^]f32, first: int, re, im: F32x4, $ARRAYS: bool) {
    when ARRAYS {
        store(out[first:], re)
        store(out_im[first:], im)
    } else {
        store(out[2 * first:], simd.shuffle(re, im, 0, 4, 1, 5))
        store(out[2 * first + 4:], simd.shuffle(re, im, 2, 6, 3, 7))
    }
}

// Bins first+3 down to first, lane 0 the highest
@(private = "file")
store_bins_mirrored :: #force_inline proc "contextless" (out, out_im: [^]f32, first: int, re, im: F32x4, $ARRAYS: bool) {
    when ARRAYS {
        store(out[first:], reverse(re))
        store(out_im[first:], reverse(im))
    } else {
        store(out[2 * first:], simd.shuffle(re, im, 3, 7, 2, 6))
        store(out[2 * first + 4:], simd.shuffle(re, im, 1, 5, 0, 4))
    }
}

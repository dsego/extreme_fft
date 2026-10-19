package extreme_fft

import "core:math/bits"


/*
    ============================================================================================
    WRITTEN BY ARTIFICIAL INTELLIGENCE: this file was written by Claude, an AI model made by
    Anthropic, in October 2026.

    Four FFTs at once, one in each SIMD lane. A twiddle is the same for all four, so every pass is
    plain vector arithmetic on whole vectors: no even and odd split of the samples, no
    transposes, no turning lanes around in the split. Those are what keeps a single transform
    from going faster, see the passes' timings in LOG.md.

    The four signals interleaved, sample n of signal t at samples[4n + t], the way four channel
    audio comes. A complex point of the half size FFT, z[k] = x[2k] + j·x[2k+1], is then the 8
    floats from 8k: the four real parts and the four imaginary parts. The work buffers and the
    bins keep that layout, bin k of the four signals at spectrum[8k] (real parts) and
    spectrum[8k + 4] (imaginary parts). Bin 0 holds DC in the real parts and Nyquist in the
    imaginary parts.

    Otherwise as in extreme_fft.odin: radix-4 Stockham stages, the last one inside the split,
    bin k and bin N/2 - k together. Forward only and not scaled, N a power of two, 64 or more.
    ============================================================================================
*/


// A plan for four transforms of one size. Not for two threads at once, the buffers are the plan's.
Lanes_Plan :: struct {
    size:        int, // N, real samples of each signal
    half:        int, // N/2, complex points and bins of each
    stages:      []Stage, // all but the last stage, see make_stages
    last_radix:  int, // 2 or 4, the last stage, run inside the split
    last_stride: int, // N/2 / last_radix
    buffers:     [2][]f32, // ping and pong, N/2 points of 8 floats each
    split_re:    []f32, // ½·W_N^k for k ≤ N/4
    split_im:    []f32,
}

create_lanes_plan :: proc(size: int, allocator := context.allocator) -> (plan: Lanes_Plan) {
    assert(bits.is_power_of_two(size) && size >= 64)
    context.allocator = allocator

    plan.size = size
    plan.half = size / 2
    for &buffer in plan.buffers do buffer = make_floats(8 * plan.half)
    plan.stages, plan.last_radix, plan.last_stride = make_stages(plan.half)
    plan.split_re, plan.split_im = make_split_twiddles(size)
    return
}

destroy_lanes_plan :: proc(plan: ^Lanes_Plan, allocator := context.allocator) {
    context.allocator = allocator
    for stage in plan.stages do delete(stage.twiddles)
    delete(plan.stages)
    for buffer in plan.buffers do delete(buffer)
    delete(plan.split_re)
    delete(plan.split_im)
}

// Four signals of N samples, interleaved, to their N/2 bins, 8 floats a bin
run_lanes :: proc(plan: ^Lanes_Plan, samples: []f32, spectrum: []f32) #no_bounds_check {
    assert(len(samples) == 4 * plan.size && len(spectrum) == 4 * plan.size)

    first_stage4(plan.stages[0], raw_data(samples), raw_data(plan.buffers[0]))
    current := 0
    for stage in plan.stages[1:] {
        radix4_stage4(stage, raw_data(plan.buffers[current]), raw_data(plan.buffers[1 - current]))
        current = 1 - current
    }
    if plan.last_radix == 2 {
        radix2_split4(plan, raw_data(plan.buffers[current]), raw_data(spectrum))
    } else {
        radix4_split4(plan, raw_data(plan.buffers[current]), raw_data(spectrum))
    }
}


// The first stage, stride 1, from the samples. Four p at a time for the twiddles: six vector
// loads from the table and each p's taken out of its lane, the inner loop over q is one point
// long at stride 1.
first_stage4 :: proc(stage: Stage, samples, dst: [^]f32) #no_bounds_check {
    m := stage.m
    tw := raw_data(stage.twiddles)
    for p := 0; p < m; p += 4 {
        w: [6]F32x4
        #unroll for i in 0 ..< 6 do w[i] = load(tw[i * m + p:])
        first_lane4(samples, dst, p, m, w, 0)
        first_lane4(samples, dst, p, m, w, 1)
        first_lane4(samples, dst, p, m, w, 2)
        first_lane4(samples, dst, p, m, w, 3)
    }
}

@(private = "file")
first_lane4 :: #force_inline proc "contextless" (samples, dst: [^]f32, p, m: int, w: [6]F32x4, $LANE: int) #no_bounds_check {
    i := p + LANE
    radix4_point4(
        samples, dst, i, m, 4 * i, 1, true,
        Twiddles{
            lane(w[0], LANE), lane(w[1], LANE), lane(w[2], LANE),
            lane(w[3], LANE), lane(w[4], LANE), lane(w[5], LANE),
        },
    )
}

// A radix-4 stage with a stride of 4 or more, one point per q under the twiddles of p
radix4_stage4 :: proc(stage: Stage, src, dst: [^]f32) #no_bounds_check {
    m := stage.m
    s := stage.stride
    quarter := s * m
    tw := raw_data(stage.twiddles)

    for q in 0 ..< s do radix4_point4(src, dst, q, quarter, q, s, false, {})
    for p in 1 ..< m {
        w := Twiddles{
            splat(tw[p]), splat(tw[m + p]),
            splat(tw[2 * m + p]), splat(tw[3 * m + p]),
            splat(tw[4 * m + p]), splat(tw[5 * m + p]),
        }
        for q in 0 ..< s do radix4_point4(src, dst, s * p + q, quarter, 4 * s * p + q, s, true, w)
    }
}

// One radix-4 butterfly of four transforms: inputs at points first_in + j·quarter, outputs at
// first_out + j·s
@(private = "file")
radix4_point4 :: #force_inline proc "contextless" (
    src, dst: [^]f32,
    first_in, quarter, first_out, s: int,
    $TWIDDLED: bool,
    w: Twiddles,
) #no_bounds_check {
    a_re, a_im := point(src, first_in)
    b_re, b_im := point(src, first_in + quarter)
    c_re, c_im := point(src, first_in + 2 * quarter)
    d_re, d_im := point(src, first_in + 3 * quarter)
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im := butterfly4(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im)
    when TWIDDLED {
        y1_re, y1_im = cmul(y1_re, y1_im, w.w1_re, w.w1_im)
        y2_re, y2_im = cmul(y2_re, y2_im, w.w2_re, w.w2_im)
        y3_re, y3_im = cmul(y3_re, y3_im, w.w3_re, w.w3_im)
    }
    set_point(dst, first_out, y0_re, y0_im)
    set_point(dst, first_out + s, y1_re, y1_im)
    set_point(dst, first_out + 2 * s, y2_re, y2_im)
    set_point(dst, first_out + 3 * s, y3_re, y3_im)
}


/*
    The split as in extreme_fft.odin, X[k] = E + T and X[N/2 - k] = conj(E - T), with Z[k] and
    Z[N/2 - k] each a point of its own: nothing to turn around, no edge bins one at a time.
*/

// The split with a radix-2 last stage, s = N/4: Z[q] = x[q] + x[q+s], Z[q+s] = x[q] - x[q+s]
radix2_split4 :: proc(plan: ^Lanes_Plan, x, out: [^]f32) #no_bounds_check {
    s := plan.last_stride
    half := plan.half // in a local like the twiddles, see radix2_split
    twiddle_re, twiddle_im := raw_data(plan.split_re), raw_data(plan.split_im)

    // DC and Nyquist out of Z[0], and bin N/4, its own mirror, out of Z[s] = x[0] - x[s]
    a_re, a_im := point(x, 0)
    b_re, b_im := point(x, s)
    z_re, z_im := a_re + b_re, a_im + b_im
    set_point(out, 0, z_re + z_im, z_re - z_im)
    z_re, z_im = a_re - b_re, a_im - b_im
    split_pair4(out, twiddle_re, twiddle_im, half, s, z_re, z_im, z_re, z_im)

    // Z[k] = x[k] + x[k+s] and Z[N/2 - k] = Z[s + (s - k)] = x[s-k] - x[2s-k]
    for k in 1 ..< s {
        l_re, l_im := point(x, k)
        h_re, h_im := point(x, k + s)
        p_re, p_im := point(x, s - k)
        q_re, q_im := point(x, 2 * s - k)
        split_pair4(out, twiddle_re, twiddle_im, half, k, l_re + h_re, l_im + h_im, p_re - q_re, p_im - q_im)
    }
}

// The split with a radix-4 last stage, s = N/8. Z[q + j·s] is output j of the butterfly at q,
// and k = q + j·s pairs with N/2 - k = (s - q) + (3 - j)·s.
radix4_split4 :: proc(plan: ^Lanes_Plan, x, out: [^]f32) #no_bounds_check {
    s := plan.last_stride
    half := plan.half
    twiddle_re, twiddle_im := raw_data(plan.split_re), raw_data(plan.split_im)

    // q = 0: Z[0] is DC and Nyquist, Z[s] pairs with Z[3s], Z[2s] with itself
    {
        z0_re, z0_im, z1_re, z1_im, z2_re, z2_im, z3_re, z3_im := butterfly_at4(x, 0, s)
        set_point(out, 0, z0_re + z0_im, z0_re - z0_im)
        split_pair4(out, twiddle_re, twiddle_im, half, s, z1_re, z1_im, z3_re, z3_im)
        split_pair4(out, twiddle_re, twiddle_im, half, 2 * s, z2_re, z2_im, z2_re, z2_im)
    }

    // q and its mirror s - q, the low bin of each pair always under N/4 so the twiddle is in the
    // table: q + j·s for j = 0, 1 and (s - q) + (3 - j)·s for j = 2, 3
    for q in 1 ..< s / 2 {
        r := s - q
        q0_re, q0_im, q1_re, q1_im, q2_re, q2_im, q3_re, q3_im := butterfly_at4(x, q, s)
        r0_re, r0_im, r1_re, r1_im, r2_re, r2_im, r3_re, r3_im := butterfly_at4(x, r, s)
        split_pair4(out, twiddle_re, twiddle_im, half, q, q0_re, q0_im, r3_re, r3_im)
        split_pair4(out, twiddle_re, twiddle_im, half, q + s, q1_re, q1_im, r2_re, r2_im)
        split_pair4(out, twiddle_re, twiddle_im, half, r, r0_re, r0_im, q3_re, q3_im)
        split_pair4(out, twiddle_re, twiddle_im, half, r + s, r1_re, r1_im, q2_re, q2_im)
    }

    // q = s/2 is its own mirror: Z[s/2] pairs with Z[s/2 + 3s], Z[s/2 + s] with Z[s/2 + 2s]
    {
        q := s / 2
        z0_re, z0_im, z1_re, z1_im, z2_re, z2_im, z3_re, z3_im := butterfly_at4(x, q, s)
        split_pair4(out, twiddle_re, twiddle_im, half, q, z0_re, z0_im, z3_re, z3_im)
        split_pair4(out, twiddle_re, twiddle_im, half, q + s, z1_re, z1_im, z2_re, z2_im)
    }
}

// The last stage's butterfly at q, no twiddles
@(private = "file")
butterfly_at4 :: #force_inline proc "contextless" (x: [^]f32, q, s: int) -> (y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im: F32x4) {
    a_re, a_im := point(x, q)
    b_re, b_im := point(x, q + s)
    c_re, c_im := point(x, q + 2 * s)
    d_re, d_im := point(x, q + 3 * s)
    return butterfly4(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im)
}

// Bin k and bin N/2 - k of the four transforms, out of Z[k] (low) and Z[N/2 - k] (high)
@(private = "file")
split_pair4 :: #force_inline proc "contextless" (
    out, twiddle_re, twiddle_im: [^]f32,
    half, k: int,
    low_re, low_im, high_re, high_im: F32x4,
) #no_bounds_check {
    HALF :: F32x4{0.5, 0.5, 0.5, 0.5}
    w_re, w_im := splat(twiddle_re[k]), splat(twiddle_im[k])

    sum_re, diff_im := low_re + high_re, low_im - high_im
    diff_re, sum_im := low_re - high_re, low_im + high_im
    odd_im := mul_add(diff_re, w_im, sum_im * w_re)
    minus_odd_re := mul_add(-diff_re, w_re, sum_im * w_im)
    even_re := HALF * sum_re

    x_im := mul_add(HALF, diff_im, minus_odd_re)
    set_point(out, k, even_re + odd_im, x_im)
    set_point(out, half - k, even_re - odd_im, x_im - diff_im) // see split_four
}

// Point k of the four transforms, its real and imaginary parts
@(private = "file")
point :: #force_inline proc "contextless" (p: [^]f32, k: int) -> (re, im: F32x4) {
    return load(p[8 * k:]), load(p[8 * k + 4:])
}

@(private = "file")
set_point :: #force_inline proc "contextless" (p: [^]f32, k: int, re, im: F32x4) {
    store(p[8 * k:], re)
    store(p[8 * k + 4:], im)
}

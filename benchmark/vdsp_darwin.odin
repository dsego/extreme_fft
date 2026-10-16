package benchmark

// Apple's FFTs in Accelerate's vDSP. The _darwin suffix keeps this file out of other builds.

foreign import accelerate "system:Accelerate.framework"

DSPSplitComplex :: struct {
    realp: [^]f32,
    imagp: [^]f32,
}

FFT_RADIX_2 :: 0
FFT_FORWARD :: 1
DFT_FORWARD :: 1

@(default_calling_convention = "c")
foreign accelerate {
    vDSP_create_fftsetup :: proc(log2n: uint, radix: i32) -> rawptr ---
    vDSP_destroy_fftsetup :: proc(setup: rawptr) ---
    vDSP_fft_zrip :: proc(setup: rawptr, c: ^DSPSplitComplex, stride: int, log2n: uint, direction: i32) ---
    vDSP_ctoz :: proc(c: [^]f32, ic: int, z: ^DSPSplitComplex, iz: int, n: uint) ---

    vDSP_DFT_zrop_CreateSetup :: proc(previous: rawptr, length: uint, direction: i32) -> rawptr ---
    vDSP_DFT_Execute :: proc(setup: rawptr, ir, ii, or_, oi: [^]f32) ---
    vDSP_DFT_DestroySetup :: proc(setup: rawptr) ---
}

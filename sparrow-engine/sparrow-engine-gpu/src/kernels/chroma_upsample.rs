//! libjpeg-turbo-exact chroma upsampling + YCbCr -> RGB for nvjpeg YUV output.
//!
//! See `chroma_upsample.cu` for the arithmetic. The kernel is compiled once
//! per process and loaded once per CUDA device, so every nvjpeg decode site
//! (the stateless `decode_jpeg` path and the per-model `JpegDecoder`s) shares
//! it without owning a handle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx, Ptx};
use sparrow_engine_types::error::{Result, SparrowEngineError};

const KERNEL_SRC: &str = include_str!("chroma_upsample.cu");
const KERNEL_NAME: &str = "chroma_upsample_ycc_rgb";

/// Chroma layout of the planes handed to [`chroma_upsample_gpu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaLayout {
    /// 4:2:0 — chroma halved horizontally and vertically (libjpeg h2v2).
    H2V2,
    /// 4:2:2 — chroma halved horizontally only (libjpeg h2v1).
    H2V1,
}

fn kernel_for(stream: &Arc<CudaStream>) -> Result<CudaFunction> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    static FUNCS: OnceLock<Mutex<HashMap<usize, CudaFunction>>> = OnceLock::new();

    let ctx = stream.context();
    let ordinal = ctx.ordinal();
    let funcs = FUNCS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = funcs
        .lock()
        .map_err(|_| SparrowEngineError::Ort("chroma_upsample kernel cache poisoned".into()))?;
    if let Some(f) = guard.get(&ordinal) {
        return Ok(f.clone());
    }
    let ptx = PTX
        .get_or_init(|| compile_ptx(KERNEL_SRC).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile chroma_upsample.cu: {e}")))?;
    let module = ctx
        .load_module(ptx.clone())
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module chroma_upsample: {e}")))?;
    let func = module.load_function(KERNEL_NAME).map_err(|e| {
        SparrowEngineError::Ort(format!("cudarc load_function chroma_upsample: {e}"))
    })?;
    guard.insert(ordinal, func.clone());
    Ok(func)
}

/// Upsample planar Y/Cb/Cr (chroma at native subsampling) into an HWC RGB
/// u8 buffer, bit-matching libjpeg-turbo's fancy upsampling and colour
/// conversion. `y` is `w x h` with pitch `y_pitch`; `cb`/`cr` are `cw x ch`
/// with pitch `c_pitch`.
#[allow(clippy::too_many_arguments)]
pub fn chroma_upsample_gpu(
    stream: &Arc<CudaStream>,
    y: &CudaSlice<u8>,
    y_pitch: usize,
    cb: &CudaSlice<u8>,
    cr: &CudaSlice<u8>,
    c_pitch: usize,
    cw: u32,
    ch: u32,
    w: u32,
    h: u32,
    layout: ChromaLayout,
) -> Result<CudaSlice<u8>> {
    let func = kernel_for(stream)?;
    let to_i32 = |v: usize, what: &str| {
        i32::try_from(v).map_err(|_| SparrowEngineError::ImageDecode(format!("{what} exceeds i32")))
    };
    let y_pitch_i = to_i32(y_pitch, "luma pitch")?;
    let c_pitch_i = to_i32(c_pitch, "chroma pitch")?;
    let cw_i = to_i32(cw as usize, "chroma width")?;
    let ch_i = to_i32(ch as usize, "chroma height")?;
    let w_i = to_i32(w as usize, "width")?;
    let h_i = to_i32(h as usize, "height")?;
    let mode: i32 = match layout {
        ChromaLayout::H2V2 => 0,
        ChromaLayout::H2V1 => 1,
    };
    // libjpeg-turbo jinit_upsampler only selects fancy upsampling when the
    // downsampled width exceeds 2; narrower components are replicated.
    let fancy: i32 = i32::from(cw > 2);

    let mut out: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(w as usize * h as usize * 3)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros: {e}")))?;
    const TX: u32 = 32;
    const TY: u32 = 8;
    let cfg = LaunchConfig {
        grid_dim: (w.div_ceil(TX), h.div_ceil(TY), 1),
        block_dim: (TX, TY, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&func);
    launch
        .arg(y)
        .arg(&y_pitch_i)
        .arg(cb)
        .arg(cr)
        .arg(&c_pitch_i)
        .arg(&cw_i)
        .arg(&ch_i)
        .arg(&mut out)
        .arg(&w_i)
        .arg(&h_i)
        .arg(&mode)
        .arg(&fancy);
    // SAFETY: argument list matches the kernel signature; every buffer is
    // sized for the indices the kernel reads or writes (bounds checked
    // against w/h and clamped to cw/ch inside the kernel).
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch chroma_upsample: {e}")))?;
    Ok(out)
}

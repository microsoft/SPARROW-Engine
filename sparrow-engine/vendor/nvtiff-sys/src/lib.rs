//! Minimal bindings for NVIDIA nvTIFF (`nvtiff.h`, 0.8 or newer), covering
//! single-image decode only. The library is loaded on first use, so it is an
//! optional runtime dependency.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// 0 is success.
pub type nvtiffStatus_t = c_int;
pub type nvtiffStream_t = *mut c_void;
pub type nvtiffDecoder_t = *mut c_void;
pub type nvtiffDecodeParams_t = *mut c_void;
pub type cudaStream_t = *mut c_void;

/// `nvtiffOutputFormat_t` value for interleaved 8-bit RGB output.
pub const NVTIFF_OUTPUT_RGB_I_UINT8: c_int = 1;
pub const MAX_NUM_SAMPLES: usize = 16;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nvtiffImageInfo_t {
    pub image_type: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub compression: c_int,
    pub photometric_int: c_int,
    pub planar_config: c_int,
    pub samples_per_pixel: u16,
    pub bits_per_pixel: u16,
    pub bits_per_sample: [u16; MAX_NUM_SAMPLES],
    pub sample_format: [c_int; MAX_NUM_SAMPLES],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct nvtiffImage_t {
    pub plane_data: *mut *mut c_void,
    pub plane_pitch_bytes: *mut usize,
    pub num_planes: u32,
}

pub struct NvtiffLib {
    _lib: libloading::Library,
    pub nvtiffStreamOpen: unsafe extern "C" fn(
        buffer: *const u8,
        buffer_size: usize,
        tiff_stream: *mut nvtiffStream_t,
    ) -> nvtiffStatus_t,
    pub nvtiffStreamClose: unsafe extern "C" fn(tiff_stream: nvtiffStream_t) -> nvtiffStatus_t,
    pub nvtiffStreamGetImageInfo: unsafe extern "C" fn(
        tiff_stream: nvtiffStream_t,
        ifd_offset: usize,
        image_info: *mut nvtiffImageInfo_t,
    ) -> nvtiffStatus_t,
    pub nvtiffDecoderCreateSimple:
        unsafe extern "C" fn(decoder: *mut nvtiffDecoder_t, stream: cudaStream_t) -> nvtiffStatus_t,
    pub nvtiffDecoderDestroy:
        unsafe extern "C" fn(decoder: nvtiffDecoder_t, stream: cudaStream_t) -> nvtiffStatus_t,
    pub nvtiffDecodeParamsCreate:
        unsafe extern "C" fn(decode_params: *mut nvtiffDecodeParams_t) -> nvtiffStatus_t,
    pub nvtiffDecodeParamsDestroy:
        unsafe extern "C" fn(decode_params: nvtiffDecodeParams_t) -> nvtiffStatus_t,
    pub nvtiffDecodeParamsSetOutputFormat:
        unsafe extern "C" fn(decode_params: nvtiffDecodeParams_t, format: c_int) -> nvtiffStatus_t,
    pub nvtiffDecode: unsafe extern "C" fn(
        tiff_stream: nvtiffStream_t,
        decoder: nvtiffDecoder_t,
        params: nvtiffDecodeParams_t,
        image_out: *mut nvtiffImage_t,
        stream: cudaStream_t,
    ) -> nvtiffStatus_t,
}

const ENV_VAR: &str = "SPARROW_ENGINE_NVTIFF_LIBRARY_PATH";
const DEFAULT_LIBRARY: &str = if cfg!(windows) {
    "nvtiff_0.dll"
} else {
    "libnvtiff.so.0"
};

static LIB: OnceLock<Result<NvtiffLib, String>> = OnceLock::new();

/// Returns nvTIFF, loading it on first call from
/// `SPARROW_ENGINE_NVTIFF_LIBRARY_PATH` or else the default library search
/// path. The outcome, including a load error, is cached for the process.
pub fn nvtiff() -> Result<&'static NvtiffLib, &'static str> {
    LIB.get_or_init(load).as_ref().map_err(String::as_str)
}

fn load() -> Result<NvtiffLib, String> {
    let path = std::env::var_os(ENV_VAR)
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| DEFAULT_LIBRARY.into());
    // SAFETY: nvTIFF has no load-time initialisers with preconditions.
    let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| {
        format!(
            "{} could not be loaded ({e}). GPU TIFF decoding needs nvTIFF 0.8 or newer for \
             CUDA 12: pip install nvidia-nvtiff-cu12 or install it from NVIDIA, \
             or set {ENV_VAR} to the library's absolute path.",
            path.to_string_lossy()
        )
    })?;
    // SAFETY: each field type matches its declaration in nvtiff.h. Older nvTIFF
    // releases lack nvtiffStreamOpen and the nvtiffDecode signature used here,
    // so they fail to load instead of being called with the wrong ABI.
    unsafe {
        Ok(NvtiffLib {
            nvtiffStreamOpen: sym(&lib, "nvtiffStreamOpen")?,
            nvtiffStreamClose: sym(&lib, "nvtiffStreamClose")?,
            nvtiffStreamGetImageInfo: sym(&lib, "nvtiffStreamGetImageInfo")?,
            nvtiffDecoderCreateSimple: sym(&lib, "nvtiffDecoderCreateSimple")?,
            nvtiffDecoderDestroy: sym(&lib, "nvtiffDecoderDestroy")?,
            nvtiffDecodeParamsCreate: sym(&lib, "nvtiffDecodeParamsCreate")?,
            nvtiffDecodeParamsDestroy: sym(&lib, "nvtiffDecodeParamsDestroy")?,
            nvtiffDecodeParamsSetOutputFormat: sym(&lib, "nvtiffDecodeParamsSetOutputFormat")?,
            nvtiffDecode: sym(&lib, "nvtiffDecode")?,
            _lib: lib,
        })
    }
}

/// # Safety
/// `T` must be the function pointer type of the C function `name`.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T, String> {
    unsafe { lib.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|e| format!("nvTIFF lacks {name} ({e}); version 0.8 or newer is required"))
}

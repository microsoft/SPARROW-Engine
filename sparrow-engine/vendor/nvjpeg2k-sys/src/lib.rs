//! Minimal bindings for NVIDIA nvJPEG2000 (`nvjpeg2k.h`, 0.8 or newer),
//! covering single-image decode only. The library is loaded on first use, so
//! it is an optional runtime dependency.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

/// 0 is success.
pub type nvjpeg2kStatus_t = c_int;
pub type nvjpeg2kHandle_t = *mut c_void;
pub type nvjpeg2kDecodeState_t = *mut c_void;
pub type nvjpeg2kStream_t = *mut c_void;
pub type nvjpeg2kDecodeParams_t = *mut c_void;
pub type cudaStream_t = *mut c_void;

/// `nvjpeg2kImageType_t` value for 8-bit unsigned samples.
pub const NVJPEG2K_UINT8: c_int = 0;
/// `nvjpeg2kImageFormat_t` value for interleaved (HWC) output.
pub const NVJPEG2K_FORMAT_INTERLEAVED: c_int = 1;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nvjpeg2kImageInfo_t {
    pub image_width: u32,
    pub image_height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
    pub num_components: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct nvjpeg2kImageComponentInfo_t {
    pub component_width: u32,
    pub component_height: u32,
    pub precision: u8,
    pub sgn: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct nvjpeg2kImage_t {
    pub pixel_data: *mut *mut c_void,
    pub pitch_in_bytes: *mut usize,
    pub pixel_type: c_int,
    pub num_components: u32,
}

pub struct Nvjpeg2kLib {
    _lib: libloading::Library,
    pub nvjpeg2kCreateSimple:
        unsafe extern "C" fn(handle: *mut nvjpeg2kHandle_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDestroy: unsafe extern "C" fn(handle: nvjpeg2kHandle_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeStateCreate: unsafe extern "C" fn(
        handle: nvjpeg2kHandle_t,
        decode_state: *mut nvjpeg2kDecodeState_t,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeStateDestroy:
        unsafe extern "C" fn(decode_state: nvjpeg2kDecodeState_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kStreamCreate:
        unsafe extern "C" fn(stream_handle: *mut nvjpeg2kStream_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kStreamDestroy:
        unsafe extern "C" fn(stream_handle: nvjpeg2kStream_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kStreamParse: unsafe extern "C" fn(
        handle: nvjpeg2kHandle_t,
        data: *const u8,
        length: usize,
        save_metadata: c_int,
        save_stream: c_int,
        stream_handle: nvjpeg2kStream_t,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kStreamGetImageInfo: unsafe extern "C" fn(
        stream_handle: nvjpeg2kStream_t,
        image_info: *mut nvjpeg2kImageInfo_t,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kStreamGetImageComponentInfo: unsafe extern "C" fn(
        stream_handle: nvjpeg2kStream_t,
        component_info: *mut nvjpeg2kImageComponentInfo_t,
        component_id: u32,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeParamsCreate:
        unsafe extern "C" fn(decode_params: *mut nvjpeg2kDecodeParams_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeParamsDestroy:
        unsafe extern "C" fn(decode_params: nvjpeg2kDecodeParams_t) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeParamsSetRGBOutput: unsafe extern "C" fn(
        decode_params: nvjpeg2kDecodeParams_t,
        force_rgb: i32,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeParamsSetOutputFormat: unsafe extern "C" fn(
        decode_params: nvjpeg2kDecodeParams_t,
        format: c_int,
    ) -> nvjpeg2kStatus_t,
    pub nvjpeg2kDecodeImage: unsafe extern "C" fn(
        handle: nvjpeg2kHandle_t,
        decode_state: nvjpeg2kDecodeState_t,
        jpeg2k_stream: nvjpeg2kStream_t,
        decode_params: nvjpeg2kDecodeParams_t,
        decode_output: *mut nvjpeg2kImage_t,
        stream: cudaStream_t,
    ) -> nvjpeg2kStatus_t,
}

const ENV_VAR: &str = "SPARROW_ENGINE_NVJPEG2K_LIBRARY_PATH";
const DEFAULT_LIBRARY: &str = if cfg!(windows) {
    "nvjpeg2k_0.dll"
} else {
    "libnvjpeg2k.so.0"
};

static LIB: OnceLock<Result<Nvjpeg2kLib, String>> = OnceLock::new();

/// Returns nvJPEG2000, loading it on first call from
/// `SPARROW_ENGINE_NVJPEG2K_LIBRARY_PATH` or else the default library search
/// path. The outcome, including a load error, is cached for the process.
pub fn nvjpeg2k() -> Result<&'static Nvjpeg2kLib, &'static str> {
    LIB.get_or_init(load).as_ref().map_err(String::as_str)
}

fn load() -> Result<Nvjpeg2kLib, String> {
    let path = std::env::var_os(ENV_VAR)
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| DEFAULT_LIBRARY.into());
    // SAFETY: nvJPEG2000 has no load-time initialisers with preconditions.
    let lib = unsafe { libloading::Library::new(&path) }.map_err(|e| {
        format!(
            "{} could not be loaded ({e}). GPU JPEG 2000 decoding needs nvJPEG2000 0.8 or newer \
             for CUDA 12: pip install nvidia-nvjpeg2k-cu12 or install it from NVIDIA, \
             or set {ENV_VAR} to the library's absolute path.",
            path.to_string_lossy()
        )
    })?;
    // SAFETY: each field type matches its declaration in nvjpeg2k.h.
    unsafe {
        Ok(Nvjpeg2kLib {
            nvjpeg2kCreateSimple: sym(&lib, "nvjpeg2kCreateSimple")?,
            nvjpeg2kDestroy: sym(&lib, "nvjpeg2kDestroy")?,
            nvjpeg2kDecodeStateCreate: sym(&lib, "nvjpeg2kDecodeStateCreate")?,
            nvjpeg2kDecodeStateDestroy: sym(&lib, "nvjpeg2kDecodeStateDestroy")?,
            nvjpeg2kStreamCreate: sym(&lib, "nvjpeg2kStreamCreate")?,
            nvjpeg2kStreamDestroy: sym(&lib, "nvjpeg2kStreamDestroy")?,
            nvjpeg2kStreamParse: sym(&lib, "nvjpeg2kStreamParse")?,
            nvjpeg2kStreamGetImageInfo: sym(&lib, "nvjpeg2kStreamGetImageInfo")?,
            nvjpeg2kStreamGetImageComponentInfo: sym(&lib, "nvjpeg2kStreamGetImageComponentInfo")?,
            nvjpeg2kDecodeParamsCreate: sym(&lib, "nvjpeg2kDecodeParamsCreate")?,
            nvjpeg2kDecodeParamsDestroy: sym(&lib, "nvjpeg2kDecodeParamsDestroy")?,
            nvjpeg2kDecodeParamsSetRGBOutput: sym(&lib, "nvjpeg2kDecodeParamsSetRGBOutput")?,
            nvjpeg2kDecodeParamsSetOutputFormat: sym(&lib, "nvjpeg2kDecodeParamsSetOutputFormat")?,
            nvjpeg2kDecodeImage: sym(&lib, "nvjpeg2kDecodeImage")?,
            _lib: lib,
        })
    }
}

/// # Safety
/// `T` must be the function pointer type of the C function `name`.
unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T, String> {
    unsafe { lib.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|e| format!("nvJPEG2000 lacks {name} ({e}); version 0.8 or newer is required"))
}

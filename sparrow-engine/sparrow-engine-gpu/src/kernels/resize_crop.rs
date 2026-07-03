//! Rust wrapper around `resize_crop.cu`. Fused resize_crop preprocess:
//! optional pre-crop-square -> separable-convolution resize (selectable
//! interpolation) -> optional center-crop to input_size -> per-channel
//! normalize -> NCHW. Mirrors `sparrow-engine-cpu`'s `resize_crop` (manifest
//! method "resize_crop"); the crop / center-crop parameters are derived here
//! exactly as the CPU derives them (pre_crop_square, resize_mode, center_crop).
//!
//! ENG-RESIZE Phase 2: the GPU counterpart of the CPU `resize_crop` shipped in
//! `feat(cpu): resize_crop preprocessing method` (ONB-1 center-crop
//! classifiers: awc135, the YOLOv8-cls trio, nz-species, queensland).

use std::sync::Arc;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ChannelOrder, Interpolation, ResizeCropConfig, ResizeMode};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::decode::GpuImage;
use crate::kernels::tiled_preprocess::NormalizeStats;

const KERNEL_SRC: &str = include_str!("resize_crop.cu");
const KERNEL_NAME: &str = "resize_crop_kernel";

#[derive(Clone)]
pub struct ResizeCropKernel {
    func: CudaFunction,
}

impl ResizeCropKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile resize_crop.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module resize_crop: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("cudarc load_function resize_crop: {e}"))
            })?;
        Ok(Self { func })
    }
}

/// Resolved integer geometry for the resize_crop pipeline. Derived from the
/// manifest [`ResizeCropConfig`] + source dims + model `input_size` with the
/// SAME arithmetic as `sparrow-engine-cpu`'s `resize_crop`.
struct ResizeCropGeom {
    crop_x: i32,
    crop_y: i32,
    crop_w: i32,
    crop_h: i32,
    rw: i32,
    rh: i32,
    off_x: i32,
    off_y: i32,
}

fn resolve_geometry(
    src_w: u32,
    src_h: u32,
    rc: &ResizeCropConfig,
    input_size: [u32; 2],
) -> Result<ResizeCropGeom> {
    let (tgt_w, tgt_h) = (input_size[0], input_size[1]);

    // 1. Optional center-square pre-crop (Ultralytics / alita).
    let (crop_x, crop_y, crop_w, crop_h) = if rc.pre_crop_square {
        let m = src_w.min(src_h);
        ((src_w - m) / 2, (src_h - m) / 2, m, m)
    } else {
        (0, 0, src_w, src_h)
    };

    // 2. Resize target (rw, rh).
    let (rw, rh) = match rc.resize_mode {
        ResizeMode::Exact => (rc.resize_size[0], rc.resize_size[1]),
        ResizeMode::ShorterSide => {
            let s = rc.resize_size[0] as f32;
            let (w, h) = (crop_w as f32, crop_h as f32);
            let scale = s / w.min(h);
            (
                (w * scale).round().max(1.0) as u32,
                (h * scale).round().max(1.0) as u32,
            )
        }
    };

    // 3. Optional center-crop to input_size (exact pixel slice).
    let (off_x, off_y) = if rc.center_crop {
        if rw < tgt_w || rh < tgt_h {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "resize_crop: resized {rw}x{rh} is smaller than center_crop target {tgt_w}x{tgt_h}"
            )));
        }
        ((rw - tgt_w) / 2, (rh - tgt_h) / 2)
    } else {
        if rw != tgt_w || rh != tgt_h {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "resize_crop produced {rw}x{rh} but model input_size is {tgt_w}x{tgt_h} \
                 (set center_crop=true, or resize_size to match input_size)"
            )));
        }
        (0, 0)
    };

    Ok(ResizeCropGeom {
        crop_x: crop_x as i32,
        crop_y: crop_y as i32,
        crop_w: crop_w as i32,
        crop_h: crop_h as i32,
        rw: rw as i32,
        rh: rh as i32,
        off_x: off_x as i32,
        off_y: off_y as i32,
    })
}

/// Fused resize_crop preprocess on GPU. Output is (tgt_w × tgt_h × 3) NCHW f32.
///
/// `input_size` = model input dims (final output). `rc` + `interp` come from
/// the manifest. Honours `channel_order` (RGB / BGR) and `stats`
/// (Unit / ImageNet) exactly like [`super::resize::resize_gpu`].
#[allow(clippy::too_many_arguments)]
pub fn resize_crop_gpu(
    stream: &Arc<CudaStream>,
    kernel: &ResizeCropKernel,
    src: &GpuImage,
    rc: &ResizeCropConfig,
    input_size: [u32; 2],
    channel_order: ChannelOrder,
    stats: NormalizeStats,
    interp: Interpolation,
) -> Result<CudaSlice<f32>> {
    let (tgt_w, tgt_h) = (input_size[0], input_size[1]);
    let geom = resolve_geometry(src.width, src.height, rc, input_size)?;

    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    let mut dst: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).map_err(|e| {
        SparrowEngineError::Ort(format!("cudarc alloc_zeros (resize_crop dst): {e}"))
    })?;

    let bgr_flag: i32 = match channel_order {
        ChannelOrder::Rgb => 0,
        ChannelOrder::Bgr => 1,
    };
    let interp_flag: i32 = match interp {
        Interpolation::Bilinear => 0,
        Interpolation::Bicubic => 1,
        Interpolation::Lanczos => 2,
    };
    let unit_flag: i32 = if stats == NormalizeStats::UNIT { 1 } else { 0 };

    const TX: u32 = 16;
    const TY: u32 = 16;
    let cfg = LaunchConfig {
        grid_dim: (tgt_w.div_ceil(TX), tgt_h.div_ceil(TY), 1),
        block_dim: (TX, TY, 1),
        shared_mem_bytes: 0,
    };

    let src_w_i = src.width as i32;
    let src_h_i = src.height as i32;
    let tgt_w_i = tgt_w as i32;
    let tgt_h_i = tgt_h as i32;
    let mean_r = stats.mean[0];
    let mean_g = stats.mean[1];
    let mean_b = stats.mean[2];
    let std_r = stats.std[0];
    let std_g = stats.std[1];
    let std_b = stats.std[2];

    let mut launch = stream.launch_builder(&kernel.func);
    launch
        .arg(&src.data)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&geom.crop_x)
        .arg(&geom.crop_y)
        .arg(&geom.crop_w)
        .arg(&geom.crop_h)
        .arg(&geom.rw)
        .arg(&geom.rh)
        .arg(&geom.off_x)
        .arg(&geom.off_y)
        .arg(&mean_r)
        .arg(&mean_g)
        .arg(&mean_b)
        .arg(&std_r)
        .arg(&std_g)
        .arg(&std_b)
        .arg(&unit_flag)
        .arg(&bgr_flag)
        .arg(&interp_flag);

    // SAFETY: kernel signature matches args; bounds check inside kernel.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch resize_crop_kernel: {e}")))?;

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::GpuImage;
    use image::{Rgb, RgbImage};

    fn cuda_or_skip(name: &str) -> Option<Arc<CudaContext>> {
        if std::env::var("SPARROW_ENGINE_GPU_TESTS").as_deref() == Ok("0") {
            eprintln!("SPARROW_ENGINE_GPU_TESTS=0 -> skipping {name}");
            return None;
        }
        match CudaContext::new(0) {
            Ok(c) => Some(c),
            Err(_) => {
                eprintln!("CUDA unavailable -> skipping {name}");
                None
            }
        }
    }

    fn synthetic(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let r = ((x * 17 + y * 5) % 256) as u8;
            let g = ((x * 3 + y * 29) % 256) as u8;
            let b = (((x ^ y) * 11) % 256) as u8;
            Rgb([r, g, b])
        })
    }

    // CPU reference replicating sparrow-engine-cpu::preprocess::resize_crop
    // (crop_imm -> image resize -> crop_imm) then unit /255 -> NCHW.
    fn cpu_ref_nchw(
        img: &RgbImage,
        rc: &ResizeCropConfig,
        input_size: [u32; 2],
        filter: image::imageops::FilterType,
    ) -> Vec<f32> {
        let (tw, th) = (input_size[0], input_size[1]);
        let base: RgbImage = if rc.pre_crop_square {
            let m = img.width().min(img.height());
            let x = (img.width() - m) / 2;
            let y = (img.height() - m) / 2;
            image::imageops::crop_imm(img, x, y, m, m).to_image()
        } else {
            img.clone()
        };
        let (rw, rh) = match rc.resize_mode {
            ResizeMode::Exact => (rc.resize_size[0], rc.resize_size[1]),
            ResizeMode::ShorterSide => {
                let s = rc.resize_size[0] as f32;
                let (w, h) = (base.width() as f32, base.height() as f32);
                let scale = s / w.min(h);
                (
                    (w * scale).round().max(1.0) as u32,
                    (h * scale).round().max(1.0) as u32,
                )
            }
        };
        let resized = image::imageops::resize(&base, rw, rh, filter);
        let final_img: RgbImage = if rc.center_crop {
            let x = (resized.width() - tw) / 2;
            let y = (resized.height() - th) / 2;
            image::imageops::crop_imm(&resized, x, y, tw, th).to_image()
        } else {
            resized
        };
        let plane = (tw * th) as usize;
        let mut out = vec![0f32; 3 * plane];
        for y in 0..th {
            for x in 0..tw {
                let p = final_img.get_pixel(x, y);
                let idx = (y * tw + x) as usize;
                out[idx] = p[0] as f32 / 255.0;
                out[plane + idx] = p[1] as f32 / 255.0;
                out[2 * plane + idx] = p[2] as f32 / 255.0;
            }
        }
        out
    }

    fn run_case(
        name: &str,
        sw: u32,
        sh: u32,
        rc: ResizeCropConfig,
        input_size: [u32; 2],
        interp: Interpolation,
        filter: image::imageops::FilterType,
    ) {
        let ctx = match cuda_or_skip(name) {
            Some(c) => c,
            None => return,
        };
        let stream = ctx.default_stream();
        let kernel = ResizeCropKernel::new(&ctx).expect("compile resize_crop kernel");
        let img = synthetic(sw, sh);
        let data = stream.clone_htod(&img.as_raw().clone()).expect("htod");
        let gpu_img = GpuImage {
            data,
            width: sw,
            height: sh,
        };
        let dev = resize_crop_gpu(
            &stream,
            &kernel,
            &gpu_img,
            &rc,
            input_size,
            ChannelOrder::Rgb,
            NormalizeStats::UNIT,
            interp,
        )
        .expect("resize_crop_gpu");
        let got: Vec<f32> = stream.clone_dtoh(&dev).expect("dtoh");
        stream.synchronize().expect("sync");
        let want = cpu_ref_nchw(&img, &rc, input_size, filter);
        assert_eq!(got.len(), want.len());
        let mut maxd = 0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            maxd = maxd.max((a - b).abs());
        }
        assert!(
            maxd < 2.0 / 255.0,
            "{name}: max abs diff {maxd} vs CPU resize_crop exceeds 2/255"
        );
    }

    // Ultralytics YOLOv8-cls idiom: pre-crop square + exact resize (bilinear).
    #[test]
    fn resize_crop_gpu_ultralytics_exact_bilinear() {
        run_case(
            "ultralytics_exact_bilinear",
            50,
            40,
            ResizeCropConfig {
                pre_crop_square: true,
                resize_size: [24, 24],
                resize_mode: ResizeMode::Exact,
                center_crop: false,
            },
            [24, 24],
            Interpolation::Bilinear,
            image::imageops::FilterType::Triangle,
        );
    }

    // torchvision Resize(shorter)+CenterCrop idiom (awc135): bicubic.
    #[test]
    fn resize_crop_gpu_shorter_side_center_crop_bicubic() {
        run_case(
            "shorter_side_center_crop_bicubic",
            60,
            44,
            ResizeCropConfig {
                pre_crop_square: false,
                resize_size: [28, 0],
                resize_mode: ResizeMode::ShorterSide,
                center_crop: true,
            },
            [20, 20],
            Interpolation::Bicubic,
            image::imageops::FilterType::CatmullRom,
        );
    }

    // alita / nz-species idiom: square crop + lanczos resize + center-crop.
    #[test]
    fn resize_crop_gpu_square_lanczos_center_crop() {
        run_case(
            "square_lanczos_center_crop",
            58,
            46,
            ResizeCropConfig {
                pre_crop_square: true,
                resize_size: [30, 30],
                resize_mode: ResizeMode::Exact,
                center_crop: true,
            },
            [24, 24],
            Interpolation::Lanczos,
            image::imageops::FilterType::Lanczos3,
        );
    }
}

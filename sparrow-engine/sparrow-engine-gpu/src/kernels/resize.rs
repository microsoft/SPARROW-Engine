//! Direct GPU resize, per-channel normalization, and NCHW transpose.
//!
//! Classifiers opt into the CPU/PIL-style final u8 rounding before
//! normalization. Other convolution-resize callers retain floating-point
//! pixels, including float-tensor `ResizeMinMax` detectors and encoders.
//!
//! Reuses the `NormalizeStats` enum already defined in
//! `sparrow_engine_gpu::kernels::tiled_preprocess` — same `(mean, std)` shape used
//! by HerdNet / OWL-T tiled preprocess.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ChannelOrder, Interpolation};

use crate::decode::GpuImage;
use crate::kernels::tiled_preprocess::NormalizeStats;

const KERNEL_SRC: &str = include_str!("resize.cu");
const KERNEL_NAME: &str = "resize_kernel";

#[derive(Clone, Copy)]
enum ResizeRounding {
    PreserveFloat,
    RoundToU8,
}

#[derive(Clone)]
pub struct ResizeKernel {
    func: CudaFunction,
}

impl ResizeKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile resize.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module resize: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function resize: {e}")))?;
        Ok(Self { func })
    }
}

/// Convolutional bilinear resize + per-channel normalize + NCHW preprocess on GPU.
///
/// Direct resize from `src` (HWC u8 RGB) to (tgt_w × tgt_h × 3) NCHW f32,
/// no aspect preservation, no crop. Honours the manifest `channel_order`:
/// RGB → plane order [R, G, B]; BGR → plane order [B, G, R].
///
/// Convolution filters preserve fractional pixels. The cv2 interpolation
/// mode retains its existing integer conversion. Classifiers that resize
/// u8 images use [`resize_classifier_gpu`] instead.
///
/// `stats` controls the post-resize normalization. `NormalizeStats::UNIT`
/// reproduces the pre-Amazon `/255` behaviour bit-exactly.
///
#[allow(clippy::too_many_arguments)]
pub fn resize_gpu(
    stream: &Arc<CudaStream>,
    kernel: &ResizeKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    channel_order: ChannelOrder,
    stats: NormalizeStats,
    interp: Interpolation,
) -> Result<CudaSlice<f32>> {
    resize_gpu_with_rounding(
        stream,
        kernel,
        src,
        tgt_w,
        tgt_h,
        channel_order,
        stats,
        interp,
        ResizeRounding::PreserveFloat,
    )
}

/// Direct classifier resize with final u8 rounding before normalization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resize_classifier_gpu(
    stream: &Arc<CudaStream>,
    kernel: &ResizeKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    channel_order: ChannelOrder,
    stats: NormalizeStats,
    interp: Interpolation,
) -> Result<CudaSlice<f32>> {
    resize_gpu_with_rounding(
        stream,
        kernel,
        src,
        tgt_w,
        tgt_h,
        channel_order,
        stats,
        interp,
        ResizeRounding::RoundToU8,
    )
}

#[allow(clippy::too_many_arguments)]
fn resize_gpu_with_rounding(
    stream: &Arc<CudaStream>,
    kernel: &ResizeKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    channel_order: ChannelOrder,
    stats: NormalizeStats,
    interp: Interpolation,
    rounding: ResizeRounding,
) -> Result<CudaSlice<f32>> {
    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    let mut dst: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (resize dst): {e}")))?;

    let bgr_flag: i32 = match channel_order {
        ChannelOrder::Rgb => 0,
        ChannelOrder::Bgr => 1,
    };

    const TX: u32 = 16;
    const TY: u32 = 16;
    let cfg = LaunchConfig {
        grid_dim: (tgt_w.div_ceil(TX), tgt_h.div_ceil(TY), 1),
        block_dim: (TX, TY, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernel.func);
    let src_w_i: i32 = src.width as i32;
    let src_h_i: i32 = src.height as i32;
    let tgt_w_i: i32 = tgt_w as i32;
    let tgt_h_i: i32 = tgt_h as i32;
    let mean_r = stats.mean[0];
    let mean_g = stats.mean[1];
    let mean_b = stats.mean[2];
    let std_r = stats.std[0];
    let std_g = stats.std[1];
    let std_b = stats.std[2];
    // Unit-identity guard: when stats == NormalizeStats::UNIT, take the
    // fast-path branch in the kernel that emits `px/255` directly.
    // Belt-and-braces against any FMA / contraction-rewrite the NVRTC
    // backend might apply to the general `(px/255 - 0)/1` form. The
    // comparison is exact-equal against the public UNIT constants
    // (mean=[0,0,0], std=[1,1,1]) — any other Unit-equivalent stats
    // (e.g., mean=[0,0,0], std=[1.0,1.0,1.0001]) take the general path,
    // which is the correct behaviour.
    let unit_flag: i32 = if stats == NormalizeStats::UNIT { 1 } else { 0 };
    // Interpolation filter selector — mirrors sparrow-engine-cpu's resize dispatch:
    // Bilinear -> Triangle, Bicubic -> CatmullRom, Lanczos -> Lanczos3,
    // Cv2Bilinear -> cv2 INTER_LINEAR fixed 2x2.
    let interp_flag: i32 = match interp {
        Interpolation::Nearest => {
            return Err(SparrowEngineError::InvalidManifest(
                "nearest classifier resize is routed through the shared host reference path"
                    .to_string(),
            ));
        }
        Interpolation::Bilinear => 0,
        Interpolation::Bicubic => 1,
        Interpolation::Lanczos => 2,
        Interpolation::Cv2Bilinear => 3,
    };
    let round_u8_flag: i32 = match rounding {
        ResizeRounding::PreserveFloat => 0,
        ResizeRounding::RoundToU8 => 1,
    };

    launch
        .arg(&src.data)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&mean_r)
        .arg(&mean_g)
        .arg(&mean_b)
        .arg(&std_r)
        .arg(&std_g)
        .arg(&std_b)
        .arg(&unit_flag)
        .arg(&bgr_flag)
        .arg(&interp_flag)
        .arg(&round_u8_flag);

    // SAFETY: kernel signature matches args; bounds check inside kernel.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch resize_kernel: {e}")))?;

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

    // High-frequency synthetic image so bilinear / bicubic / lanczos produce
    // meaningfully DIFFERENT outputs — a filter-routing bug (always bilinear)
    // would then fail the bicubic / lanczos cases.
    fn synthetic(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let r = ((x * 17 + y * 5) % 256) as u8;
            let g = ((x * 3 + y * 29) % 256) as u8;
            let b = (((x ^ y) * 11) % 256) as u8;
            Rgb([r, g, b])
        })
    }

    // CPU reference: the exact production resize (image crate, u8) -> unit /255
    // -> NCHW, matching sparrow-engine-cpu's resize_direct + build_tensor(unit).
    fn cpu_ref_nchw(
        img: &RgbImage,
        tw: u32,
        th: u32,
        filter: image::imageops::FilterType,
    ) -> Vec<f32> {
        let resized = image::imageops::resize(img, tw, th, filter);
        let plane = (tw * th) as usize;
        let mut out = vec![0f32; 3 * plane];
        for y in 0..th {
            for x in 0..tw {
                let p = resized.get_pixel(x, y);
                let idx = (y * tw + x) as usize;
                out[idx] = p[0] as f32 / 255.0;
                out[plane + idx] = p[1] as f32 / 255.0;
                out[2 * plane + idx] = p[2] as f32 / 255.0;
            }
        }
        out
    }

    fn cpu_ref_cv2_nchw(img: &RgbImage, tw: u32, th: u32) -> Vec<f32> {
        let scale_x = img.width() as f32 / tw as f32;
        let scale_y = img.height() as f32 / th as f32;
        let plane = (tw * th) as usize;
        let mut out = vec![0f32; 3 * plane];
        for oy in 0..th {
            for ox in 0..tw {
                let src_x = (ox as f32 + 0.5) * scale_x - 0.5;
                let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
                let x0f = src_x.floor();
                let y0f = src_y.floor();
                let fx = src_x - x0f;
                let fy = src_y - y0f;
                let x0 = (x0f as i32).clamp(0, img.width() as i32 - 1) as u32;
                let x1 = (x0f as i32 + 1).clamp(0, img.width() as i32 - 1) as u32;
                let y0 = (y0f as i32).clamp(0, img.height() as i32 - 1) as u32;
                let y1 = (y0f as i32 + 1).clamp(0, img.height() as i32 - 1) as u32;
                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);
                let idx = (oy * tw + ox) as usize;
                for (c, plane_offset) in [0, plane, 2 * plane].into_iter().enumerate() {
                    let top = p00[c] as f32 * (1.0 - fx) + p10[c] as f32 * fx;
                    let bottom = p01[c] as f32 * (1.0 - fx) + p11[c] as f32 * fx;
                    out[plane_offset + idx] =
                        (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) / 255.0;
                }
            }
        }
        out
    }

    fn run_case(name: &str, interp: Interpolation, filter: image::imageops::FilterType) {
        let ctx = match cuda_or_skip(name) {
            Some(c) => c,
            None => return,
        };
        let stream = ctx.default_stream();
        let kernel = ResizeKernel::new(&ctx).expect("compile resize kernel");
        // Non-integer downsample ratio exercises multi-tap windows.
        let (sw, sh, tw, th) = (40u32, 32u32, 17u32, 13u32);
        let img = synthetic(sw, sh);
        let host_rgb: Vec<u8> = img.as_raw().clone();
        let data = stream.clone_htod(&host_rgb).expect("htod");
        let gpu_img = GpuImage {
            data,
            width: sw,
            height: sh,
        };
        let dev = resize_gpu(
            &stream,
            &kernel,
            &gpu_img,
            tw,
            th,
            ChannelOrder::Rgb,
            NormalizeStats::UNIT,
            interp,
        )
        .expect("resize_gpu");
        let got: Vec<f32> = stream.clone_dtoh(&dev).expect("dtoh");
        stream.synchronize().expect("sync");
        let want = cpu_ref_nchw(&img, tw, th, filter);
        assert_eq!(got.len(), want.len());
        let mut maxd = 0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            maxd = maxd.max((a - b).abs());
        }
        // GPU keeps f32 (clamped, no u8 round); CPU rounds to u8 -> <=0.5/255
        // rounding gap + float ULP. 2/255 headroom still catches a wrong filter
        // (bilinear vs bicubic diverge far more than that on this image).
        assert!(
            maxd < 2.0 / 255.0,
            "{name}: max abs diff {maxd} vs image-crate {filter:?} exceeds 2/255"
        );
    }

    #[test]
    fn classifier_rounds_half_pixels_before_normalization_and_channel_order() {
        let Some(ctx) = cuda_or_skip("classifier_rounding_boundary") else {
            return;
        };
        let stream = ctx.default_stream();
        let kernel = ResizeKernel::new(&ctx).expect("compile resize kernel");
        let image = RgbImage::from_raw(2, 1, vec![0, 10, 254, 1, 11, 255]).unwrap();
        let gpu_image = GpuImage {
            data: stream.clone_htod(image.as_raw()).expect("htod"),
            width: 2,
            height: 1,
        };
        let rounded = image::imageops::resize(&image, 1, 1, image::imageops::FilterType::Triangle);
        assert_eq!(rounded.get_pixel(0, 0).0, [1, 11, 255]);
        for stats in [
            NormalizeStats::UNIT,
            NormalizeStats::IMAGENET,
            NormalizeStats::RAW,
        ] {
            for order in [ChannelOrder::Rgb, ChannelOrder::Bgr] {
                let quantized = resize_classifier_gpu(
                    &stream,
                    &kernel,
                    &gpu_image,
                    1,
                    1,
                    order,
                    stats,
                    Interpolation::Bilinear,
                )
                .expect("classifier resize");
                let floating = resize_gpu(
                    &stream,
                    &kernel,
                    &gpu_image,
                    1,
                    1,
                    order,
                    stats,
                    Interpolation::Bilinear,
                )
                .expect("float resize");
                let got = stream.clone_dtoh(&quantized).expect("dtoh");
                let kept = stream.clone_dtoh(&floating).expect("dtoh");
                for plane in 0..3 {
                    let channel = if order == ChannelOrder::Rgb {
                        plane
                    } else {
                        2 - plane
                    };
                    let normalize =
                        |pixel: f32| (pixel / 255.0 - stats.mean[channel]) / stats.std[channel];
                    let expected = normalize(rounded.get_pixel(0, 0).0[channel] as f32);
                    let fractional = normalize([0.5, 10.5, 254.5][channel]);
                    let ulps = 4.0 * f32::EPSILON * expected.abs().max(1.0);
                    assert!((got[plane] - expected).abs() <= ulps);
                    assert!((kept[plane] - fractional).abs() <= ulps);
                    assert!((got[plane] - kept[plane]).abs() > 1e-4);
                }
            }
        }
    }

    #[test]
    fn classifier_convolution_filters_quantize_but_default_resize_keeps_fractions() {
        let Some(ctx) = cuda_or_skip("classifier_convolution_rounding") else {
            return;
        };
        let stream = ctx.default_stream();
        let kernel = ResizeKernel::new(&ctx).expect("compile resize kernel");
        let image = synthetic(40, 32);
        let gpu_image = GpuImage {
            data: stream.clone_htod(image.as_raw()).expect("htod"),
            width: 40,
            height: 32,
        };
        for interpolation in [
            Interpolation::Bilinear,
            Interpolation::Bicubic,
            Interpolation::Lanczos,
        ] {
            let quantized = resize_classifier_gpu(
                &stream,
                &kernel,
                &gpu_image,
                17,
                13,
                ChannelOrder::Rgb,
                NormalizeStats::UNIT,
                interpolation,
            )
            .expect("classifier resize");
            let floating = resize_gpu(
                &stream,
                &kernel,
                &gpu_image,
                17,
                13,
                ChannelOrder::Rgb,
                NormalizeStats::UNIT,
                interpolation,
            )
            .expect("float resize");
            let got = stream.clone_dtoh(&quantized).expect("dtoh");
            let kept = stream.clone_dtoh(&floating).expect("dtoh");
            assert!(got.iter().all(|value| {
                let pixel = value * 255.0;
                (0.0..=255.0).contains(&pixel)
                    && (pixel - pixel.round()).abs() <= 4.0 * f32::EPSILON * 255.0
            }));
            assert!(kept.iter().any(|value| {
                let pixel = value * 255.0;
                (pixel - pixel.round()).abs() > 0.1
            }));
        }
    }

    #[test]
    fn resize_gpu_matches_image_crate_bilinear() {
        run_case(
            "bilinear",
            Interpolation::Bilinear,
            image::imageops::FilterType::Triangle,
        );
    }

    #[test]
    fn resize_gpu_matches_image_crate_bicubic() {
        run_case(
            "bicubic",
            Interpolation::Bicubic,
            image::imageops::FilterType::CatmullRom,
        );
    }

    #[test]
    fn resize_gpu_matches_image_crate_lanczos() {
        run_case(
            "lanczos",
            Interpolation::Lanczos,
            image::imageops::FilterType::Lanczos3,
        );
    }

    #[test]
    fn resize_gpu_matches_cv2_bilinear() {
        let ctx = match cuda_or_skip("cv2_bilinear") {
            Some(c) => c,
            None => return,
        };
        let stream = ctx.default_stream();
        let kernel = ResizeKernel::new(&ctx).expect("compile resize kernel");
        let (sw, sh, tw, th) = (40u32, 32u32, 17u32, 13u32);
        let img = synthetic(sw, sh);
        let host_rgb: Vec<u8> = img.as_raw().clone();
        let data = stream.clone_htod(&host_rgb).expect("htod");
        let gpu_img = GpuImage {
            data,
            width: sw,
            height: sh,
        };
        let dev = resize_gpu(
            &stream,
            &kernel,
            &gpu_img,
            tw,
            th,
            ChannelOrder::Rgb,
            NormalizeStats::UNIT,
            Interpolation::Cv2Bilinear,
        )
        .expect("resize_gpu");
        let got: Vec<f32> = stream.clone_dtoh(&dev).expect("dtoh");
        stream.synchronize().expect("sync");
        let want = cpu_ref_cv2_nchw(&img, tw, th);
        assert_eq!(got.len(), want.len());
        let maxd = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // CUDA arithmetic can differ by one final u8 rounding step versus the
        // CPU f32 reference; larger gaps indicate a coordinate/filter mismatch.
        assert!(
            maxd <= 1.0 / 255.0,
            "cv2_bilinear max abs diff {maxd} exceeds 1/255"
        );
    }
}

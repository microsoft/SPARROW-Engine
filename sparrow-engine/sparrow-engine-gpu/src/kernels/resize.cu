// CUDA kernel: convolutional bilinear resize + per-channel normalize + NCHW
// transpose. Bit-tight against `fast_image_resize::Resizer` with
// `ResizeAlg::Convolution(FilterType::Bilinear)` — the exact algorithm
// `sparrow-engine-cpu/src/preprocess.rs::resize_simd` uses. Replaces the earlier
// 2-tap texture-style bilinear (which differed at the bilinear-impl LSB
// from the CPU baseline and flipped 1/10 borderline-confidence top-1
// labels in the SpeciesNet parity test).
//
// Algorithm (matches `fast_image_resize-5.5.0/src/convolution/mod.rs::
// precompute_coefficients` + `bilinear_filter`):
//   scale       = in_size  / out_size       (per axis)
//   filter_scale = max(1.0, scale)           (adaptive AA, on for Convolution())
//   filter_radius = 1.0 * filter_scale       (Bilinear support = 1.0)
//   For each out pixel:
//     in_center  = (out + 0.5) * scale
//     center     = in_center - 0.5
//     window     = [floor(in_center - radius), ceil(in_center + radius))
//                  clamped to [0, in_size)
//     For each x in window:
//       t = (x - center) / filter_scale
//       w = max(0, 1 - |t|)          // Bilinear filter (triangle / tent)
//     Normalize weights so sum == 1.0.
//   Output pixel = Σ_x Σ_y w_x[x] * w_y[y] * src[y, x] (separable conv)
//
// Window size:
//   window_size = ceil(filter_radius) * 2 + 1
//   For SpeciesNet 1280×960 → 480×480 (~2.67× downsample): window = 7
//   For 4× downsample: window = 9. We size the on-stack weight arrays
//   to 16 (covers up to ~7× downsample) and clamp via WMAX to keep the
//   kernel branch-free.
//
// Normalization (post-resize):
//   out = (px / 255 - mean[c]) / std[c]
//   Unit (SpeciesNet):     mean = [0, 0, 0], std = [1, 1, 1] — identity.
//   ImageNet (Amazon CTV2): mean = [0.485, 0.456, 0.406],
//                            std = [0.229, 0.224, 0.225].
//
// Unit-identity guard: an explicit `unit_norm` flag (computed in resize.rs
// from the NormalizeStats values) selects a fast path that emits `px/255`
// directly without the subtract+divide step. Two reasons:
//   1. Belt-and-braces against any FMA / contraction-rewrite the NVRTC
//      backend might introduce — IEEE 754 says `x-0=x` and `x/1=x` are
//      exact ops, but the fast path is a true byte-identity skip rather
//      than relying on optimizer fidelity.
//   2. Skips two redundant arithmetic ops in the hot loop for the common
//      Unit case (SpeciesNet today, possibly more classifiers later).
// The result for the Unit case is bit-tight against the pre-2026-05-03
// `/255`-only kernel and against sparrow-engine-cpu's `resize_direct` reference.
//
// Inputs:
//   src       — HWC u8 RGB device buffer of size (src_h * src_w * 3).
//                Always RGB — channel-order swap happens at output write.
//   src_w     — source image width  (pixels).
//   src_h     — source image height (pixels).
//   dst       — NCHW f32 output of size (3 * tgt_h * tgt_w).
//   tgt_w     — target tensor width.
//   tgt_h     — target tensor height.
//   mean_*    — per-channel mean for normalization in [0,1] space.
//   std_*     — per-channel std  for normalization.
//   unit_norm — 1 → Unit fast path (out = px/255, byte-identity skip).
//                0 → general path (out = (px/255 - mean) / std).
//                The wrapper `resize_gpu` sets this when stats == UNIT.
//   bgr       — 0 → RGB plane order. 1 → BGR plane order.

// Separable-convolution resize + per-channel normalize + NCHW transpose.
// Matches `image` 0.25.10 `imageops::resize` (the algorithm
// `sparrow-engine-cpu/src/preprocess.rs::resize_pil` uses via
// `image::imageops::resize`). Selectable interpolation (ENG-RESIZE Phase 2):
//   interp = 0 → Triangle   (bilinear, support 1)  — PIL BILINEAR
//   interp = 1 → CatmullRom (bicubic,  support 2)  — PIL BICUBIC (b=0, c=0.5)
//   interp = 2 → Lanczos3   (support 3)            — PIL LANCZOS
//
// image crate resize is a two-pass separable filter with a full-precision f32
// intermediate (`vertical_sample -> Rgba32FImage -> horizontal_sample`), so a
// single-pass 2D separable convolution with the SAME per-axis normalized
// weights is mathematically equivalent (differences are float-ordering ULPs,
// << 1/255). Per-axis weights (matching image crate horizontal_sample):
//   ratio        = in_size / out_size            (per axis)
//   sratio       = max(1.0, ratio)               (adaptive AA)
//   in_center    = (out + 0.5) * ratio
//   center       = in_center - 0.5
//   src_support  = support * sratio              (support: 1 / 2 / 3)
//   window       = [clamp(floor(in_center - src_support), 0, in-1),
//                   clamp(ceil (in_center + src_support), left+1, in))
//   weight(x)    = kernel((x - center) / sratio)   then normalize so Σ = 1
//   out pixel    = Σ_x Σ_y wx[x] * wy[y] * src[y, x]  (separable)
//
// Weights are computed WITHOUT a fixed-size on-stack array (a two-pass
// sum-then-accumulate) so arbitrarily large windows (high-downsample Lanczos3)
// are exact — no truncation. The kernel functions are evaluated at the
// sratio-scaled coordinate `t`, matching image crate's `(i - center)/sratio`.
//
// interp = 0 (bilinear) is byte-identical to the pre-ENG-RESIZE-Phase-2 kernel:
// support 1 → src_support == sratio == the old `radius`; the per-tap weight
// `wx*wy` and the `acc += w*p` order are unchanged.
//
// Normalization (post-resize) + Unit-identity guard + channel order are
// unchanged from the original kernel (see below).

// Windowed-sinc (Lanczos) building block: normalized sinc, sinc(0)=1.
__device__ __forceinline__ float se_sinc(float x) {
    if (x == 0.0f) return 1.0f;
    float a = x * 3.14159265358979323846f;
    return __sinf(a) / a;
}

// Filter kernel evaluated at the sratio-scaled coordinate `t`. Mirrors the
// `image` crate kernels: triangle_kernel, catmullrom_kernel (bc_cubic_spline
// b=0 c=0.5), lanczos3_kernel (lanczos with t=3).
__device__ __forceinline__ float se_filter(float t, int interp) {
    float a = fabsf(t);
    if (interp == 0) {
        // Triangle (bilinear), support 1.0.
        return a < 1.0f ? 1.0f - a : 0.0f;
    } else if (interp == 1) {
        // Catmull-Rom cubic spline (b=0, c=0.5), support 2.0. image crate
        // bc_cubic_spline: k/6 with the (b=0,c=0.5)-specialized coefficients.
        float k;
        if (a < 1.0f) {
            k = 9.0f * a * a * a - 15.0f * a * a + 6.0f;
        } else if (a < 2.0f) {
            k = -3.0f * a * a * a + 15.0f * a * a - 24.0f * a + 12.0f;
        } else {
            return 0.0f;
        }
        return k / 6.0f;
    } else {
        // Lanczos3, support 3.0: sinc(t) * sinc(t/3) for |t| < 3.
        if (a < 3.0f) {
            return se_sinc(t) * se_sinc(t / 3.0f);
        }
        return 0.0f;
    }
}

// Filter support radius (in output-normalized units) per interp mode.
__device__ __forceinline__ float se_support(int interp) {
    return interp == 0 ? 1.0f : (interp == 1 ? 2.0f : 3.0f);
}

extern "C" __global__ void resize_kernel(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    int unit_norm,
    int bgr,
    int interp
) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= tgt_w || oy >= tgt_h) return;

    int plane_size = tgt_w * tgt_h;
    int idx = oy * tgt_w + ox;

    float scale_x = (float)src_w / (float)tgt_w;
    float scale_y = (float)src_h / (float)tgt_h;
    float fscale_x = fmaxf(1.0f, scale_x);
    float fscale_y = fmaxf(1.0f, scale_y);
    float recip_x = 1.0f / fscale_x;
    float recip_y = 1.0f / fscale_y;
    float support = se_support(interp);
    float radius_x = support * fscale_x;
    float radius_y = support * fscale_y;

    float in_cx = ((float)ox + 0.5f) * scale_x;
    float in_cy = ((float)oy + 0.5f) * scale_y;
    float center_x = in_cx - 0.5f;
    float center_y = in_cy - 0.5f;

    // Window bounds — image crate horizontal_sample:
    //   left  = clamp(floor(in_center - src_support), 0, in-1)
    //   right = clamp(ceil (in_center + src_support), left+1, in)
    int xmin = (int)floorf(in_cx - radius_x);
    if (xmin < 0) xmin = 0;
    if (xmin > src_w - 1) xmin = src_w - 1;
    int xmax = (int)ceilf(in_cx + radius_x);
    if (xmax < xmin + 1) xmax = xmin + 1;
    if (xmax > src_w) xmax = src_w;

    int ymin = (int)floorf(in_cy - radius_y);
    if (ymin < 0) ymin = 0;
    if (ymin > src_h - 1) ymin = src_h - 1;
    int ymax = (int)ceilf(in_cy + radius_y);
    if (ymax < ymin + 1) ymax = ymin + 1;
    if (ymax > src_h) ymax = src_h;

    int n_x = xmax - xmin;
    int n_y = ymax - ymin;

    // Per-axis weight sums (no on-stack array → arbitrary window sizes, exact
    // for high-downsample Lanczos3). Weight arg is the sratio-scaled coordinate
    // `(src - center)/fscale`, matching image crate's `(i - center)/sratio`.
    float wx_sum = 0.0f;
    for (int i = 0; i < n_x; i++) {
        wx_sum += se_filter(((float)(xmin + i) - center_x) * recip_x, interp);
    }
    float wy_sum = 0.0f;
    for (int j = 0; j < n_y; j++) {
        wy_sum += se_filter(((float)(ymin + j) - center_y) * recip_y, interp);
    }
    float inv_wx = wx_sum != 0.0f ? 1.0f / wx_sum : 0.0f;
    float inv_wy = wy_sum != 0.0f ? 1.0f / wy_sum : 0.0f;

    // Separable convolution. Per-tap weight (fx/Σfx)*(fy/Σfy) reproduces the
    // image crate's separately-normalized wx[i]*wy[j]; the j-outer / i-inner
    // order + `acc += w*p` match the pre-Phase-2 bilinear kernel exactly, so
    // interp=0 output is byte-identical for realistic inputs.
    float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
    int row_stride = src_w * 3;
    for (int j = 0; j < n_y; j++) {
        int sy = ymin + j;
        const unsigned char* row = src + sy * row_stride;
        float wj = se_filter(((float)(ymin + j) - center_y) * recip_y, interp) * inv_wy;
        for (int i = 0; i < n_x; i++) {
            int sx = xmin + i;
            const unsigned char* p = row + sx * 3;
            float wx_i = se_filter(((float)(xmin + i) - center_x) * recip_x, interp) * inv_wx;
            float w = wx_i * wj;
            r_acc += w * (float)p[0];
            g_acc += w * (float)p[1];
            b_acc += w * (float)p[2];
        }
    }

    // Clamp the resized pixel to [0, 255] before normalize — matches the
    // `image` crate's final u8 conversion `FloatNearest(clamp(t, 0, 255))`
    // (the GPU skips the u8 round, keeping f32; the ≤0.5/255 rounding gap is
    // the established pre-Phase-2 CPU↔GPU tolerance). No-op for bilinear
    // (non-negative weights sum to 1 → convex, always in range); load-bearing
    // for CatmullRom / Lanczos3 whose negative lobes ring past [0, 255].
    r_acc = fminf(255.0f, fmaxf(0.0f, r_acc));
    g_acc = fminf(255.0f, fmaxf(0.0f, g_acc));
    b_acc = fminf(255.0f, fmaxf(0.0f, b_acc));

    // /255 + per-channel (mean, std).
    // Unit fast path (`unit_norm == 1`): out = px / 255 directly. This is
    // a true byte-identity skip vs the pre-2026-05-03 kernel — no FMA /
    // contraction concerns, no reliance on optimizer fidelity. SpeciesNet
    // uses this branch (manifest normalization = "unit").
    // General path: out = (px / 255 - mean) / std. ImageNet-normalized
    // classifiers (Amazon CTV2 today) take this branch.
    float r, g, b;
    if (unit_norm) {
        r = r_acc / 255.0f;
        g = g_acc / 255.0f;
        b = b_acc / 255.0f;
    } else {
        r = (r_acc / 255.0f - mean_r) / std_r;
        g = (g_acc / 255.0f - mean_g) / std_g;
        b = (b_acc / 255.0f - mean_b) / std_b;
    }

    if (bgr == 0) {
        dst[0 * plane_size + idx] = r;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = b;
    } else {
        dst[0 * plane_size + idx] = b;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = r;
    }
}

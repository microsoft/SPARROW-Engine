// CUDA kernel: letterbox (aspect-preserving resize + pad) + /255 normalize
// + NCHW transpose. Bit-tight against `fast_image_resize::Resizer` with
// `ResizeAlg::Convolution(FilterType::Bilinear)` — the exact algorithm
// `sparrow-engine-cpu/src/preprocess.rs::resize_simd` uses for its letterbox path.
//
// Wave 2 amend (2026-05-03): replaces the earlier 2-tap texture-style
// bilinear with the convolutional bilinear matching coder-w3's
// `kernels/resize.cu` (commit 3c4a680). The 2-tap variant produced
// detection-count drift +3/100 on DeepFaune at scale=0.75 (Triangle
// vs 2-tap divergence at non-identity scales). Multi-tap is bit-tight
// against the CPU pipeline's `fast_image_resize::Resizer(Bilinear)`.
//
// Algorithm (matches `fast_image_resize-5.5.0/src/convolution/mod.rs::
// precompute_coefficients` + `bilinear_filter`):
//   per_axis_scale = src_axis_size / new_axis_size   (per axis)
//   filter_scale   = max(1.0, per_axis_scale)        (adaptive AA when downsampling)
//   filter_radius  = 1.0 * filter_scale              (Bilinear support = 1.0)
//   For each output (resized) pixel:
//     in_center  = (out + 0.5) * per_axis_scale
//     center     = in_center - 0.5
//     window     = [floor(in_center - radius), ceil(in_center + radius))
//                  clamped to [0, src_axis_size)
//     For each x in window:
//       t = (x - center) / filter_scale
//       w = max(0, 1 - |t|)          // Bilinear filter (triangle / tent)
//     Normalize weights so sum == 1.0.
//   Output pixel = Σ_x Σ_y w_x[x] * w_y[y] * src[y, x] (separable conv)
//
// At scale=1.0 (MDv6 1280×960 → 1280 letterbox: per_axis_scale = 1.0):
// the kernel reduces to picking the source pixel directly (window of 3
// taps with center weight = 1.0; outer weights = 0 after triangle
// evaluation at distance 1.0). Identity-equivalent — MDv6 numbers
// preserved.
//
// Window size:
//   window_size = ceil(filter_radius) * 2 + 1
//   For DeepFaune 1280×960 → 960 letterbox (per_axis_scale = 1.333):
//     ceil(1.333) * 2 + 1 = 5 (so up to 25 source-pixel reads / output).
//   For 4× downsample: window = 9. WMAX=16 covers up to ~7× downsample.
//
// Inputs:
//   src       — HWC u8 RGB device buffer of size (src_h * src_w * 3).
//                Always RGB — channel-order swap happens at output write.
//   src_w     — source image width  (pixels).
//   src_h     — source image height (pixels).
//   dst       — NCHW f32 output of size (3 * tgt_h * tgt_w). Pre-zeroed by
//                caller; padding regions are filled here with `pad_value`.
//   tgt_w     — target tensor width  (e.g. 1280 for MDv6).
//   tgt_h     — target tensor height (e.g. 1280 for MDv6).
//   new_w     — resized image width  inside the canvas, = round(src_w * scale).
//   new_h     — resized image height inside the canvas, = round(src_h * scale).
//   pad_x     — left padding column index (= floor((tgt_w - new_w)/2)).
//   pad_y     — top  padding row index    (= ceil ((tgt_h - new_h)/2)).
//                NOTE: ceil to match PW preprocess (extra pixel on TOP).
//   scale     — image-space → resized-space scale factor. Unused by the
//                multi-tap kernel itself (which derives per-axis ratios
//                from src_w/new_w and src_h/new_h directly), but kept in
//                the parameter list so the Rust wrapper can pass it
//                without changes for back-compat with Wave 1.
//   pad_value — value placed into padding region after /255 normalize.
//   bgr       — 0 → emit RGB plane order (plane 0 = R).
//                1 → emit BGR plane order (plane 0 = B).
//   cv2       — 1 → cv2 INTER_LINEAR non-antialiased fixed 2x2 bilinear;
//                0 → historical anti-aliased Triangle filter.

#define WMAX 16

extern "C" __global__ void letterbox_kernel(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    float scale,
    float pad_value,
    int bgr,
    int cv2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= tgt_w || y >= tgt_h) return;

    int plane_size = tgt_w * tgt_h;
    int idx = y * tgt_w + x;

    int xi = x - pad_x;
    int yi = y - pad_y;

    float r, g, b;
    if (xi < 0 || yi < 0 || xi >= new_w || yi >= new_h) {
        // Pad region — already-normalized value.
        r = pad_value; g = pad_value; b = pad_value;
    } else {
        float scale_x = (float)src_w / (float)new_w;
        float scale_y = (float)src_h / (float)new_h;
        if (cv2) {
            float src_x = ((float)xi + 0.5f) * scale_x - 0.5f;
            float src_y = ((float)yi + 0.5f) * scale_y - 0.5f;
            float x0f = floorf(src_x);
            float y0f = floorf(src_y);
            float fx = src_x - x0f;
            float fy = src_y - y0f;

            int x0 = (int)x0f;
            if (x0 < 0) x0 = 0;
            if (x0 > src_w - 1) x0 = src_w - 1;
            int x1 = (int)x0f + 1;
            if (x1 < 0) x1 = 0;
            if (x1 > src_w - 1) x1 = src_w - 1;
            int y0 = (int)y0f;
            if (y0 < 0) y0 = 0;
            if (y0 > src_h - 1) y0 = src_h - 1;
            int y1 = (int)y0f + 1;
            if (y1 < 0) y1 = 0;
            if (y1 > src_h - 1) y1 = src_h - 1;

            int row_stride = src_w * 3;
            const unsigned char* p00 = src + y0 * row_stride + x0 * 3;
            const unsigned char* p10 = src + y0 * row_stride + x1 * 3;
            const unsigned char* p01 = src + y1 * row_stride + x0 * 3;
            const unsigned char* p11 = src + y1 * row_stride + x1 * 3;

            float r_top = (float)p00[0] * (1.0f - fx) + (float)p10[0] * fx;
            float r_bottom = (float)p01[0] * (1.0f - fx) + (float)p11[0] * fx;
            float g_top = (float)p00[1] * (1.0f - fx) + (float)p10[1] * fx;
            float g_bottom = (float)p01[1] * (1.0f - fx) + (float)p11[1] * fx;
            float b_top = (float)p00[2] * (1.0f - fx) + (float)p10[2] * fx;
            float b_bottom = (float)p01[2] * (1.0f - fx) + (float)p11[2] * fx;
            r = floorf(fminf(255.0f, fmaxf(0.0f, r_top * (1.0f - fy) + r_bottom * fy)) + 0.5f) / 255.0f;
            g = floorf(fminf(255.0f, fmaxf(0.0f, g_top * (1.0f - fy) + g_bottom * fy)) + 0.5f) / 255.0f;
            b = floorf(fminf(255.0f, fmaxf(0.0f, b_top * (1.0f - fy) + b_bottom * fy)) + 0.5f) / 255.0f;
        } else {
        // Multi-tap convolutional bilinear (Triangle filter), separable.
        // Per-axis scale = src/new (downsample > 1, upsample < 1, identity = 1).
        float fscale_x = fmaxf(1.0f, scale_x);
        float fscale_y = fmaxf(1.0f, scale_y);
        float radius_x = fscale_x; // Bilinear support = 1.0
        float radius_y = fscale_y;
        float recip_x = 1.0f / fscale_x;
        float recip_y = 1.0f / fscale_y;

        float in_cx = ((float)xi + 0.5f) * scale_x;
        float in_cy = ((float)yi + 0.5f) * scale_y;
        float center_x = in_cx - 0.5f;
        float center_y = in_cy - 0.5f;

        int xmin = (int)floorf(in_cx - radius_x);
        int xmax = (int)ceilf(in_cx + radius_x);
        int ymin = (int)floorf(in_cy - radius_y);
        int ymax = (int)ceilf(in_cy + radius_y);
        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > src_w) xmax = src_w;
        if (ymax > src_h) ymax = src_h;

        int n_x = xmax - xmin;
        int n_y = ymax - ymin;
        if (n_x <= 0 || n_y <= 0) {
            // Degenerate; emit zero (rare; should never happen for any
            // non-empty src and any (xi, yi) in [0, new_*)).
            r = 0.0f; g = 0.0f; b = 0.0f;
        } else {
            if (n_x > WMAX) n_x = WMAX;
            if (n_y > WMAX) n_y = WMAX;

            // X weights — Bilinear (triangle) filter, normalized so they sum to 1.
            float wx[WMAX];
            float wx_sum = 0.0f;
            for (int i = 0; i < n_x; i++) {
                float t = ((float)(xmin + i) - center_x) * recip_x;
                float w = 1.0f - fabsf(t);
                if (w < 0.0f) w = 0.0f;
                wx[i] = w;
                wx_sum += w;
            }
            if (wx_sum > 0.0f) {
                float inv = 1.0f / wx_sum;
                for (int i = 0; i < n_x; i++) wx[i] *= inv;
            }

            // Y weights.
            float wy[WMAX];
            float wy_sum = 0.0f;
            for (int j = 0; j < n_y; j++) {
                float t = ((float)(ymin + j) - center_y) * recip_y;
                float w = 1.0f - fabsf(t);
                if (w < 0.0f) w = 0.0f;
                wy[j] = w;
                wy_sum += w;
            }
            if (wy_sum > 0.0f) {
                float inv = 1.0f / wy_sum;
                for (int j = 0; j < n_y; j++) wy[j] *= inv;
            }

            // Separable convolution. n_x * n_y reads per output pixel.
            float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
            int row_stride = src_w * 3;
            for (int j = 0; j < n_y; j++) {
                int sy = ymin + j;
                const unsigned char* row = src + sy * row_stride;
                float wj = wy[j];
                for (int i = 0; i < n_x; i++) {
                    int sx = xmin + i;
                    const unsigned char* p = row + sx * 3;
                    float w = wx[i] * wj;
                    r_acc += w * (float)p[0];
                    g_acc += w * (float)p[1];
                    b_acc += w * (float)p[2];
                }
            }

            r = r_acc / 255.0f;
            g = g_acc / 255.0f;
            b = b_acc / 255.0f;
        }
        }
    }

    // NCHW write. Channel order at the OUTPUT only — `src` is always RGB.
    if (bgr == 0) {
        dst[0 * plane_size + idx] = r;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = b;
    } else {
        dst[0 * plane_size + idx] = b;
        dst[1 * plane_size + idx] = g;
        dst[2 * plane_size + idx] = r;
    }

    // Suppress unused-parameter warning. `scale` is part of the public
    // kernel ABI from Wave 1 and is preserved for back-compat (the Rust
    // wrapper still passes it). The multi-tap algorithm derives per-axis
    // scales from src/new directly, so this scalar is no longer needed
    // inside the kernel body.
    (void)scale;
}

// CUDA kernel: fused resize_crop preprocess — optional pre-crop-square ->
// separable-convolution resize (selectable interpolation) -> optional
// center-crop to input_size -> per-channel normalize -> NCHW transpose.
//
// Mirrors sparrow-engine-cpu/src/preprocess.rs::resize_crop (the manifest
// method "resize_crop"), which is: crop_imm(center square) -> image crate
// resize(rw, rh, filter) -> crop_imm(center input_size) -> /255 (+ norm).
//
// Fusion: each OUTPUT pixel (final input_size space) maps back through the
// center-crop (an exact pixel offset in resized space) -> the resize (a conv
// on the pre-crop base) -> the pre-crop window (an exact pixel offset in src
// space). The center-crop / pre-crop are exact integer slices, so they are
// pure coordinate offsets — only the resize does interpolation. This is
// numerically identical to the CPU crop->resize->crop chain (the CPU rounds
// the resized image to u8 before the final crop; the GPU keeps f32 and clamps
// to [0,255], the same <=0.5/255 rounding tolerance as the plain resize path).
//
// Weights + window match image 0.25.10 imageops::resize exactly (see
// resize.cu for the derivation). Weights are computed WITHOUT a fixed on-stack
// array (two-pass sum-then-accumulate) so arbitrary window sizes are exact.
//
// Coordinate mapping for output pixel (ox, oy):
//   rx = ox + off_x ;  ry = oy + off_y            (resized-space, center-crop)
//   scale = crop_w / rw  (per axis)               (resize base -> resized)
//   in_center = (r + 0.5) * scale                 (base-local center)
//   center    = in_center - 0.5
//   window    = [clamp(floor(in_center - support*fscale), 0, crop-1),
//                clamp(ceil (in_center + support*fscale), left+1, crop))
//   src read  = src[(crop_y + base_j), (crop_x + base_i)]
//
// Inputs:
//   src        — HWC u8 RGB device buffer (src_h * src_w * 3), always RGB.
//   src_w/h    — full source dims.
//   dst        — NCHW f32 output (3 * tgt_h * tgt_w).
//   tgt_w/h    — final output dims (model input_size).
//   crop_x/y   — pre-crop window origin in src space.
//   crop_w/h   — pre-crop window dims (base image the resize operates on).
//   rw/rh      — resized (intermediate) dims.
//   off_x/y    — center-crop offset in resized space ((rw-tgt_w)/2, (rh-tgt_h)/2).
//   mean_*/std_* + unit_norm — normalization (see resize.cu).
//   bgr        — 0 RGB plane order, 1 BGR.
//   interp     — 0 Triangle(bilinear) / 1 CatmullRom(bicubic) / 2 Lanczos3 /
//                3 cv2 INTER_LINEAR fixed 2x2.

__device__ __forceinline__ float rc_sinc(float x) {
    if (x == 0.0f) return 1.0f;
    float a = x * 3.14159265358979323846f;
    return __sinf(a) / a;
}

__device__ __forceinline__ float rc_filter(float t, int interp) {
    float a = fabsf(t);
    if (interp == 0) {
        return a < 1.0f ? 1.0f - a : 0.0f;
    } else if (interp == 1) {
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
        if (a < 3.0f) {
            return rc_sinc(t) * rc_sinc(t / 3.0f);
        }
        return 0.0f;
    }
}

__device__ __forceinline__ float rc_support(int interp) {
    return interp == 0 ? 1.0f : (interp == 1 ? 2.0f : 3.0f);
}

extern "C" __global__ void resize_crop_kernel(
    const unsigned char* __restrict__ src,
    int src_w,
    int src_h,
    float* __restrict__ dst,
    int tgt_w,
    int tgt_h,
    int crop_x,
    int crop_y,
    int crop_w,
    int crop_h,
    int rw,
    int rh,
    int off_x,
    int off_y,
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

    // Resized-space pixel this output maps to (exact center-crop offset).
    int rx = ox + off_x;
    int ry = oy + off_y;

    // Resize scale: pre-crop base -> resized.
    float scale_x = (float)crop_w / (float)rw;
    float scale_y = (float)crop_h / (float)rh;

    int row_stride = src_w * 3;
    float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;

    if (interp == 3) {
        float src_x = ((float)rx + 0.5f) * scale_x - 0.5f;
        float src_y = ((float)ry + 0.5f) * scale_y - 0.5f;
        float x0f = floorf(src_x);
        float y0f = floorf(src_y);
        float fx = src_x - x0f;
        float fy = src_y - y0f;

        int x0 = (int)x0f;
        if (x0 < 0) x0 = 0;
        if (x0 > crop_w - 1) x0 = crop_w - 1;
        int x1 = (int)x0f + 1;
        if (x1 < 0) x1 = 0;
        if (x1 > crop_w - 1) x1 = crop_w - 1;
        int y0 = (int)y0f;
        if (y0 < 0) y0 = 0;
        if (y0 > crop_h - 1) y0 = crop_h - 1;
        int y1 = (int)y0f + 1;
        if (y1 < 0) y1 = 0;
        if (y1 > crop_h - 1) y1 = crop_h - 1;

        const unsigned char* p00 = src + (crop_y + y0) * row_stride + (crop_x + x0) * 3;
        const unsigned char* p10 = src + (crop_y + y0) * row_stride + (crop_x + x1) * 3;
        const unsigned char* p01 = src + (crop_y + y1) * row_stride + (crop_x + x0) * 3;
        const unsigned char* p11 = src + (crop_y + y1) * row_stride + (crop_x + x1) * 3;

        float r_top = (float)p00[0] * (1.0f - fx) + (float)p10[0] * fx;
        float r_bottom = (float)p01[0] * (1.0f - fx) + (float)p11[0] * fx;
        float g_top = (float)p00[1] * (1.0f - fx) + (float)p10[1] * fx;
        float g_bottom = (float)p01[1] * (1.0f - fx) + (float)p11[1] * fx;
        float b_top = (float)p00[2] * (1.0f - fx) + (float)p10[2] * fx;
        float b_bottom = (float)p01[2] * (1.0f - fx) + (float)p11[2] * fx;
        r_acc = r_top * (1.0f - fy) + r_bottom * fy;
        g_acc = g_top * (1.0f - fy) + g_bottom * fy;
        b_acc = b_top * (1.0f - fy) + b_bottom * fy;

        r_acc = floorf(fminf(255.0f, fmaxf(0.0f, r_acc)) + 0.5f);
        g_acc = floorf(fminf(255.0f, fmaxf(0.0f, g_acc)) + 0.5f);
        b_acc = floorf(fminf(255.0f, fmaxf(0.0f, b_acc)) + 0.5f);
    } else {
    float fscale_x = fmaxf(1.0f, scale_x);
    float fscale_y = fmaxf(1.0f, scale_y);
    float recip_x = 1.0f / fscale_x;
    float recip_y = 1.0f / fscale_y;
    float support = rc_support(interp);
    float radius_x = support * fscale_x;
    float radius_y = support * fscale_y;

    // Base-local center (0-based within the pre-crop window).
    float in_cx = ((float)rx + 0.5f) * scale_x;
    float in_cy = ((float)ry + 0.5f) * scale_y;
    float center_x = in_cx - 0.5f;
    float center_y = in_cy - 0.5f;

    // Window in base-local coords, image-crate clamp; then offset into src.
    int xmin = (int)floorf(in_cx - radius_x);
    if (xmin < 0) xmin = 0;
    if (xmin > crop_w - 1) xmin = crop_w - 1;
    int xmax = (int)ceilf(in_cx + radius_x);
    if (xmax < xmin + 1) xmax = xmin + 1;
    if (xmax > crop_w) xmax = crop_w;

    int ymin = (int)floorf(in_cy - radius_y);
    if (ymin < 0) ymin = 0;
    if (ymin > crop_h - 1) ymin = crop_h - 1;
    int ymax = (int)ceilf(in_cy + radius_y);
    if (ymax < ymin + 1) ymax = ymin + 1;
    if (ymax > crop_h) ymax = crop_h;

    int n_x = xmax - xmin;
    int n_y = ymax - ymin;

    float wx_sum = 0.0f;
    for (int i = 0; i < n_x; i++) {
        wx_sum += rc_filter(((float)(xmin + i) - center_x) * recip_x, interp);
    }
    float wy_sum = 0.0f;
    for (int j = 0; j < n_y; j++) {
        wy_sum += rc_filter(((float)(ymin + j) - center_y) * recip_y, interp);
    }
    float inv_wx = wx_sum != 0.0f ? 1.0f / wx_sum : 0.0f;
    float inv_wy = wy_sum != 0.0f ? 1.0f / wy_sum : 0.0f;

    for (int j = 0; j < n_y; j++) {
        int sy = crop_y + ymin + j;
        const unsigned char* row = src + sy * row_stride;
        float wj = rc_filter(((float)(ymin + j) - center_y) * recip_y, interp) * inv_wy;
        for (int i = 0; i < n_x; i++) {
            int sx = crop_x + xmin + i;
            const unsigned char* p = row + sx * 3;
            float wx_i = rc_filter(((float)(xmin + i) - center_x) * recip_x, interp) * inv_wx;
            float w = wx_i * wj;
            r_acc += w * (float)p[0];
            g_acc += w * (float)p[1];
            b_acc += w * (float)p[2];
        }
    }

    // Clamp to [0,255] (matches image crate's final u8 conversion clamp).
    r_acc = fminf(255.0f, fmaxf(0.0f, r_acc));
    g_acc = fminf(255.0f, fmaxf(0.0f, g_acc));
    b_acc = fminf(255.0f, fmaxf(0.0f, b_acc));
    }

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

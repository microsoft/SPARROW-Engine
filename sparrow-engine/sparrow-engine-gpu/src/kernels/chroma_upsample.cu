// CUDA kernel: libjpeg-turbo "fancy" chroma upsampling + YCbCr -> RGB.
//
// nvjpeg's NVJPEG_OUTPUT_RGBI upsamples subsampled chroma by replication.
// libjpeg-turbo (PIL, OpenCV, the `image` crate's decoder) uses triangle
// ("fancy") upsampling. On small 4:2:0 images that a model then upscales,
// the difference shifts detector scores well past parity tolerance. This
// kernel takes nvjpeg's NVJPEG_OUTPUT_YUV planes and reproduces
// libjpeg-turbo's integer arithmetic exactly:
//
//   jdsample.c h2v2_fancy_upsample (4:2:0):
//     colsum(c) = 3*C[r][c] + C[rn][c], rn = r-1 (even out row) / r+1 (odd)
//     even out col: (3*colsum(c) + colsum(c-1) + 8) >> 4
//     odd  out col: (3*colsum(c) + colsum(c+1) + 7) >> 4
//   jdsample.c h2v1_fancy_upsample (4:2:2):
//     even out col: (3*C[c] + C[c-1] + 1) >> 2
//     odd  out col: (3*C[c] + C[c+1] + 2) >> 2
//   Neighbours past the edge replicate the edge sample (libjpeg context rows
//   and column special cases). libjpeg falls back to plain replication when
//   the downsampled width is <= 2; `fancy == 0` selects that.
//
//   jdcolor.c ycc_rgb_convert (SCALEBITS 16, ONE_HALF 1<<15):
//     R = Y + ((FIX(1.40200)*Cr + ONE_HALF) >> 16)
//     G = Y + ((-FIX(0.34414)*Cb + ONE_HALF - FIX(0.71414)*Cr) >> 16)
//     B = Y + ((FIX(1.77200)*Cb + ONE_HALF) >> 16)
//   with Cb, Cr centred on 128 and arithmetic right shifts.

__device__ __forceinline__ int clamp_u8(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

__device__ __forceinline__ int upsample(
    const unsigned char* p, int pitch, int cw, int ch,
    int x, int y, int mode, int fancy)
{
    int c = x >> 1;
    int r = (mode == 0) ? (y >> 1) : y;
    if (c > cw - 1) c = cw - 1;
    if (r > ch - 1) r = ch - 1;
    if (!fancy) {
        return p[r * pitch + c];
    }
    int odd_x = x & 1;
    int cn = odd_x ? min(c + 1, cw - 1) : max(c - 1, 0);
    if (mode == 0) {
        int rn = (y & 1) ? min(r + 1, ch - 1) : max(r - 1, 0);
        int s0 = 3 * p[r * pitch + c] + p[rn * pitch + c];
        int s1 = 3 * p[r * pitch + cn] + p[rn * pitch + cn];
        return (3 * s0 + s1 + (odd_x ? 7 : 8)) >> 4;
    }
    return (3 * p[r * pitch + c] + p[r * pitch + cn] + (odd_x ? 2 : 1)) >> 2;
}

extern "C" __global__ void chroma_upsample_ycc_rgb(
    const unsigned char* __restrict__ y_plane, int y_pitch,
    const unsigned char* __restrict__ cb_plane,
    const unsigned char* __restrict__ cr_plane, int c_pitch,
    int cw, int ch,
    unsigned char* __restrict__ out, int w, int h,
    int mode, int fancy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int Y = y_plane[y * y_pitch + x];
    int Cb = upsample(cb_plane, c_pitch, cw, ch, x, y, mode, fancy) - 128;
    int Cr = upsample(cr_plane, c_pitch, cw, ch, x, y, mode, fancy) - 128;

    int R = Y + ((91881 * Cr + 32768) >> 16);
    int G = Y + ((-22554 * Cb + 32768 - 46802 * Cr) >> 16);
    int B = Y + ((116130 * Cb + 32768) >> 16);

    long o = ((long)y * w + x) * 3;
    out[o + 0] = (unsigned char)clamp_u8(R);
    out[o + 1] = (unsigned char)clamp_u8(G);
    out[o + 2] = (unsigned char)clamp_u8(B);
}

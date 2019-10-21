#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "backprojection_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define Conjugate(a, b) a.x = b.x; a.y = -b.y;
#define MulReal(a,b) a.x*b.x - a.y*b.y
#define MulImag(a,b) a.x*b.y + a.y*b.x
#define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#define Multiply_Complex_Real(c,a,b) c.x = a.x * b; c.y = a.y * b
#define Multiply_Complex_Imag(c,a,b) c.x = -a.y * b; c.y = a.x * b
typedef float2 t_complex64;

constexpr int pos_dim = 2;
constexpr auto pi = 3.14159265359f;
constexpr auto c0 = 299792458.0f;

namespace tensorflow {

namespace {

__global__ void BackprojectionCUDAKernel(
          const float* pos,
          const t_complex64* data,
          t_complex64* img,
          int sweeps,
          int sweep_samples,
          float x0,
          float dx,
          int Nx,
          float y0,
          float dy,
          int Ny,
          int interp_order,
          float ref_phase,
          float v,
          float dr,
          float beamwidth,
          float tx_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= Nx || idy >= Ny) {
    return;
  }

  const float x = x0 + idx * dx;
  const float y = y0 + idy * dy;

  t_complex64 pixel = {0, 0};

  for(int i = 0; i < sweeps; i++) {
    // Sweep reference position.
    float pos_x = pos[i * pos_dim + 0];
    float pos_y = pos[i * pos_dim + 1];
    float px = (x - pos_x);
    float py = (y - pos_y);
    float target_angle = atan2(px, py);
    if (fabsf(target_angle) > beamwidth) {
        // Pixel outside of the beam.
        continue;
    }
    // Calculate distance to the pixel.
    float drx = std::sqrt(px * px + py * py);
    float dtx = std::sqrt((px + tx_offset) * (px + tx_offset) + py * py);
    float d = (drx + dtx) / 2.0f;

    t_complex64 s;
    float v_corr = -v * sinf(target_angle);
    float sx = dr * d + v_corr;
    if (interp_order == 0) {
      // Nearest neighbor.
      int id0 = lrintf(sx);
      if (id0 < 0 || id0 >= sweep_samples) {
        continue;
      }
      s = data[i * sweep_samples + id0];
    } else {
       // Linear interpolation.
       int id0 = floorf(sx);
       int id1 = id0 + 1;
       if (id0 < 0 || id1 >= sweep_samples) {
         continue;
       }
       t_complex64 s0 = data[i * sweep_samples + id0];
       t_complex64 s1 = data[i * sweep_samples + id1];

       t_complex64 st;
       Add(st, s1, -s0);
       float interp_idx = sx - id0;
       Multiply_Complex_Real(s1, st, interp_idx);
       Add(s, s0, s1);
    }
    float ref_sin, ref_cos;
    sincospif(ref_phase * d, &ref_sin, &ref_cos);
    t_complex64 ref = {ref_cos, ref_sin};
    t_complex64 s2;
    Multiply(s2, s, ref);
    Add(pixel, pixel, s2);
  }
  img[idx * Ny + idy] = pixel;
}

__global__ void BackprojectionGradCUDAKernel(
          const float* pos,
          const t_complex64* data,
          const t_complex64* grad,
          float* pos_grad,
          int sweeps,
          int sweep_samples,
          float x0,
          float dx,
          int Nx,
          float y0,
          float dy,
          int Ny,
          float ref_phase,
          float v,
          float dr,
          float beamwidth,
          float tx_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= Nx || idy >= Ny) {
    return;
  }

  const float x = x0 + idx * dx;
  const float y = y0 + idy * dy;

  t_complex64 g = grad[idx * Ny + idy];

  for(int i = 0; i < sweeps; i++) {
    // Sweep reference position.
    float pos_x = pos[i * pos_dim + 0];
    float pos_y = pos[i * pos_dim + 1];
    float px = (x - pos_x);
    float py = (y - pos_y);
    float target_angle = atan2(px, py);
    if (fabsf(target_angle) > beamwidth) {
        // Pixel outside of the beam.
        continue;
    }
    // Calculate distance to the pixel.
    float drx = std::sqrt(px * px + py * py);
    float dtx = std::sqrt((px + tx_offset) * (px + tx_offset) + py * py);
    float d = (drx + dtx) / 2.0f;

    t_complex64 s;
    float v_corr = -v * sinf(target_angle);
    float sx = dr * d + v_corr;

    // Linear interpolation.
    int id0 = floorf(sx);
    int id1 = id0 + 1;
    if (id0 < 0 || id1 >= sweep_samples) {
      continue;
    }
    t_complex64 s0 = data[i * sweep_samples + id0];
    t_complex64 s1 = data[i * sweep_samples + id1];

    t_complex64 ds;
    Add(ds, s1, -s0);

    float interp_idx = sx - id0;
    Multiply_Complex_Real(s1, ds, interp_idx);
    // Interpolated data array value
    Add(s, s0, s1);

    float ref_sin, ref_cos;
    sincospif(ref_phase * d, &ref_sin, &ref_cos);
    t_complex64 ref = {ref_cos, ref_sin};

    t_complex64 gdout, dout_conj, dout, dsum, drefs, drds;
    Multiply_Complex_Imag(drefs, s, pi * ref_phase);
    Multiply_Complex_Real(drds, ds, dr);
    Add(dsum, drefs, drds);
    Multiply(dout, ref, dsum);
    Conjugate(dout_conj, dout);
    Multiply(gdout, g, dout_conj);

    // Take real part
    float gd = gdout.x;

    float dx = 0.50f * (-px / drx - (px + tx_offset) / dtx);
    float dy = 0.50f * (-py / drx - py / dtx);
    if (!isfinite(dx)) dx = 0;
    if (!isfinite(dy)) dy = 0;

    atomicAdd(&(pos_grad[i * pos_dim + 0]), gd * dx);
    atomicAdd(&(pos_grad[i * pos_dim + 1]), gd * dy);
  }
}


}  // anonymous namespace

namespace functor {

// Define the GPU implementation that launches the CUDA kernel.
template <typename Dtype>
struct BackprojectionFunctor<GPUDevice, Dtype> {
  static void launch(const GPUDevice& d,
          const float* pos,
          const std::complex<float>* data,
          std::complex<float>* img,
          int sweeps,
          int sweep_samples,
          float x0,
          float dx,
          int Nx,
          float y0,
          float dy,
          int Ny,
          float fc,
          float v,
          float delta_r,
          float interp_order,
          float beamwidth,
          float tx_offset) {

      const auto ref_phase = 4.0f * fc / c0;
      const auto dr = 1.0f / delta_r;

      // To radians and divide by 2 to get edge angle from the center.
      beamwidth = (pi / 180.0f) * (beamwidth / 2.0f);

      unsigned int nx_block = 1;
      while (nx_block < Nx && nx_block < 256) {
        nx_block *= 2;
      }
      unsigned int ny_block = 256 / nx_block;
      dim3 thread_per_block = {nx_block, ny_block, 1};
      // Up-rounding division.
      unsigned int block_x = (Nx + thread_per_block.x - 1) / thread_per_block.x;
      unsigned int block_y = (Ny + thread_per_block.y - 1) / thread_per_block.y;
      dim3 block_count = {block_x, block_y, 1};

      BackprojectionCUDAKernel
          <<<block_count, thread_per_block, 0, d.stream()>>>(pos,
                  (t_complex64*)data, (t_complex64*)img,
                  sweeps, sweep_samples,
                  x0, dx, Nx,
                  y0, dy, Ny,
                  interp_order,
                  ref_phase, v, dr,
                  beamwidth, tx_offset);
    }
};

template struct BackprojectionFunctor<GPUDevice, float>;

// Define the GPU implementation that launches the CUDA kernel.
template <typename Dtype>
struct BackprojectionGradFunctor<GPUDevice, Dtype> {
  static void launch(const GPUDevice& d,
          const float* pos,
          const std::complex<float>* data,
          const std::complex<float>* grad,
          float* pos_grad,
          int sweeps,
          int sweep_samples,
          float x0,
          float dx,
          int Nx,
          float y0,
          float dy,
          int Ny,
          float fc,
          float v,
          float delta_r,
          float interp_order,
          float beamwidth,
          float tx_offset) {

      const auto ref_phase = 4.0f * fc / c0;
      const auto dr = 1.0f / delta_r;

      // To radians and divide by 2 to get edge angle from the center.
      beamwidth = (pi / 180.0f) * (beamwidth / 2.0f);

      unsigned int nx_block = 1;
      while (nx_block < Nx && nx_block < 256) {
        nx_block *= 2;
      }
      unsigned int ny_block = 256 / nx_block;
      dim3 thread_per_block = {nx_block, ny_block, 1};
      // Up-rounding division.
      unsigned int block_x = (Nx + thread_per_block.x - 1) / thread_per_block.x;
      unsigned int block_y = (Ny + thread_per_block.y - 1) / thread_per_block.y;
      dim3 block_count = {block_x, block_y, 1};

      cudaMemset(pos_grad, 0, pos_dim * sweeps * sizeof(float));

      BackprojectionGradCUDAKernel
          <<<block_count, thread_per_block, 0, d.stream()>>>(pos,
                  (t_complex64*)data, (t_complex64*)grad, pos_grad,
                  sweeps, sweep_samples,
                  x0, dx, Nx,
                  y0, dy, Ny,
                  ref_phase, v, dr,
                  beamwidth, tx_offset);
    }
};

template struct BackprojectionGradFunctor<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow

#endif // GOOGLE_CUDA

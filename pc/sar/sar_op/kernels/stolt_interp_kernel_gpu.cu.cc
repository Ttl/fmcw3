#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "stolt_interp_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#define Multiply_Complex_Real(c,a,b) c.x = a.x * b; c.y = a.y * b
typedef float2 t_complex64;

namespace tensorflow {

int constexpr wg_size = 4;

namespace {

// Define the CUDA kernel.
__global__ void StoltCUDAKernel(t_complex64* output_tensor,
        const t_complex64* y_tensor,
        const float* x_tensor,
        const float* x_interp_tensor,
        int y_rows, int y_cols, int x_interp_len, int order) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const float pi = 3.14159265359f;

  const int wgs = (x_interp_len + wg_size - 1) / wg_size; // Division rounds up
  const int i = index / wgs;
  const int j0 = wg_size * (index % (wgs));

  if (i >= y_rows) {
    return;
  }

  const float x_first = x_tensor[i * y_cols];
  const float x_last = x_tensor[i * y_cols  + y_cols - 1];

  int x0_i = 0;
  for (int j_wg = 0; j_wg < wg_size; j_wg++) {
    t_complex64 acc = {0.0f, 0.0f};

    int j = j0 + j_wg;
    if (j >= x_interp_len) {
        // Over the last element in column.
        return;
    }

    // Cache x_interp_tensor?
    float xi = x_interp_tensor[j];

    // Zero outside interpolation range.
    if ((xi < x_first) || (xi > x_last)) {
      output_tensor[i * x_interp_len + j] = {0.0f, 0.0f};
      continue;
    }
    // Inefficient.
    while(x_tensor[i * y_cols + x0_i] < xi && x0_i < y_cols - 1) { x0_i++; }

    for (int k = -order; k <= order; k++) {
      if (x0_i + k > y_cols - 1) {
        break;
      }
      if (x0_i + k < 0) {
        continue;
      }
      float d = 1.0f;
      if (x0_i + k != 0) {
        d = x_tensor[i * y_cols + x0_i + k] - x_tensor[i * y_cols + x0_i + k - 1];
      } else {
        d = x_tensor[i * y_cols + 1] - x_tensor[i * y_cols + 0];
      }
      float z = pi * (xi - x_tensor[i * y_cols + x0_i + k]) / d;
      float kernel = 1.0f;
      if (z != 0.0f) {
        kernel = order * sinf(z / order) * sinf(z) / (z * z);
      }
      t_complex64 acc2;
      t_complex64 v = y_tensor[i * y_cols + x0_i + k];
      Multiply_Complex_Real(acc2, v, kernel);
      Add(acc, acc, acc2);
    }
    output_tensor[i * x_interp_len + j] = acc;
  }
}

__global__ void StoltGradCUDAKernel(t_complex64* y_grad_tensor,
        const t_complex64* y_tensor,
        const float* x_tensor,
        const float* x_interp_tensor,
        t_complex64* grad_tensor,
        int y_rows, int y_cols, int x_interp_len, int order) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const float pi = 3.14159265359f;

  const int wgs = (x_interp_len + wg_size - 1) / wg_size; // Division rounds up
  const int i = index / wgs;
  const int j0 = wg_size * (index % (wgs));

  if (i >= y_rows) {
    return;
  }

  const float x_first = x_tensor[i * y_cols];
  const float x_last = x_tensor[i * y_cols  + y_cols - 1];

  int x0_i = 0;
  for (int j_wg = 0; j_wg < wg_size; j_wg++) {
    int j = j0 + j_wg;
    if (j >= x_interp_len) {
        // Over the last element in column.
        return;
    }

    // Cache x_interp_tensor?
    float xi = x_interp_tensor[j];

    // Zero outside interpolation range.
    if ((xi < x_first) || (xi > x_last)) {
      continue;
    }
    // Inefficient.
    while(x_tensor[i * y_cols + x0_i] < xi && x0_i < y_cols - 1) { x0_i++; }

    for (int k = -order; k <= order; k++) {
      if (x0_i + k > y_cols - 1) {
        break;
      }
      if (x0_i + k < 0) {
        continue;
      }
      float d = 1.0f;
      if (x0_i + k != 0) {
        d = x_tensor[i * y_cols + x0_i + k] - x_tensor[i * y_cols + x0_i + k - 1];
      } else {
        d = x_tensor[i * y_cols + 1] - x_tensor[i * y_cols + 0];
      }
      float z = pi * (xi - x_tensor[i * y_cols + x0_i + k]) / d;
      float kernel = 1.0f;
      if (z != 0.0f) {
        kernel = order * sinf(z / order) * sinf(z) / (z * z);
      }
      t_complex64 acc2;
      t_complex64 v = grad_tensor[i * x_interp_len + j];
      Multiply_Complex_Real(acc2, v, kernel);
      atomicAdd(&(y_grad_tensor[i * y_cols + x0_i + k].x), acc2.x);
      atomicAdd(&(y_grad_tensor[i * y_cols + x0_i + k].y), acc2.y);
    }
  }
}

}  // anonymous namespace

namespace functor {

// Define the GPU implementation that launches the CUDA kernel.
template <typename Dtype>
struct StoltFunctor<GPUDevice, Dtype> {
  static void launch(const GPUDevice& d,
          std::complex<float>* output_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          int y_rows, int y_cols, int x_interp_len, int order) {

      const int wgs = (x_interp_len + wg_size - 1) / wg_size;
      int thread_per_block = 256;
      int block_count = wgs * y_rows / thread_per_block;

      StoltCUDAKernel
          <<<block_count, thread_per_block, 0, d.stream()>>>((t_complex64*)output_tensor,
                  (t_complex64*)y_tensor, x_tensor, x_interp_tensor,
                  y_rows, y_cols, x_interp_len, order);
    }
};

template struct StoltFunctor<GPUDevice, float>;
//
// Define the GPU implementation that launches the CUDA kernel.
template <typename Dtype>
struct StoltGradFunctor<GPUDevice, Dtype> {
  static void launch(const GPUDevice& d,
          std::complex<float>* y_grad_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          const std::complex<float>* grad_tensor,
          int y_rows, int y_cols, int x_interp_len, int order) {

      const int wgs = (x_interp_len + wg_size - 1) / wg_size;
      int thread_per_block = 256;
      int block_count = wgs * y_rows / thread_per_block;

      cudaMemset(y_grad_tensor, 0, y_rows * y_cols * sizeof(t_complex64));

      StoltGradCUDAKernel
          <<<block_count, thread_per_block, 0, d.stream()>>>((t_complex64*)y_grad_tensor,
                  (t_complex64*)y_tensor, x_tensor, x_interp_tensor, (t_complex64*)grad_tensor,
                  y_rows, y_cols, x_interp_len, order);
    }
};

template struct StoltGradFunctor<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow

#endif // GOOGLE_CUDA

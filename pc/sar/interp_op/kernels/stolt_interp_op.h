#ifndef TFSTOLT_H_
#define TFSTOLT_H_

#include <complex>
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Dtype>
struct StoltFunctor {
  static void launch(const Device& d,
          std::complex<float>* output_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          int y_rows, int y_cols, int x_interp_len, int order);
};

template <typename Device, typename Dtype>
struct StoltGradFunctor {
  static void launch(const Device& d,
          std::complex<float>* y_grad_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          const std::complex<float>* grad_tensor,
          int y_rows, int y_cols, int x_interp_len, int order);
};

}  // namespace functor
}  // namespace tensorflow

#endif // TFSTOLT_H_

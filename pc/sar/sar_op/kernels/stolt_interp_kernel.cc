/// \file stolt_interp.cc
/// \author Henrik Forsten
/// \brief Stolt interpolation for SAR image formation.

#include "stolt_interp_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct StoltFunctor<CPUDevice, Dtype> {
  static void launch(const CPUDevice& d,
          std::complex<float>* output_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          int y_rows, int y_cols, int x_interp_len, int order) {
    constexpr auto pi = 3.14159265359f;

    auto len_x = y_cols;
    auto output_size1 = x_interp_len;

    for (int i = 0; i < y_rows; i++) {
      int x0_i = 0;
      for (int j = 0; j < output_size1; j++) {
        output_tensor[i * output_size1 + j] = 0;
        auto xi = x_interp_tensor[j];
        // Zero outside interpolation range
        if ((xi < x_tensor[i * y_cols + 0]) || (xi > x_tensor[i * y_cols + len_x-1])) {
          continue;
        }
        while(x_tensor[i * y_cols + x0_i] < xi && x0_i < len_x - 1) { x0_i++; }

        for (int k = -order; k <= order; k++) {
          if (x0_i + k > len_x - 1) {
            break;
          }
          if (x0_i + k < 0) {
            continue;
          }
          auto d = 1.0f;
          if (x0_i + k != 0) {
            d = x_tensor[i * y_cols + x0_i + k] - x_tensor[i * y_cols +  x0_i + k - 1];
          } else {
            d = x_tensor[i * y_cols + 1] - x_tensor[i * y_cols + 0];
          }
          auto z = pi * (xi - x_tensor[i * y_cols + x0_i + k]) / d;
          auto kernel = 1.0f;
          if (z != 0.0f) {
            kernel = order * std::sin(z / order) * std::sin(z) / (z * z);
          }
          output_tensor[i * x_interp_len + j] += y_tensor[i * y_cols + x0_i + k] * kernel;
        }
      }
    }
  }
};

template struct StoltFunctor<CPUDevice, float>;

template <typename Dtype>
struct StoltGradFunctor<CPUDevice, Dtype> {
  static void launch(const CPUDevice& d,
          std::complex<float>* y_grad_tensor,
          const std::complex<float>* y_tensor,
          const float* x_tensor,
          const float* x_interp_tensor,
          const std::complex<float>* grad_tensor,
          int y_rows, int y_cols, int x_interp_len, int order) {
    constexpr auto pi = 3.14159265359f;

    auto len_x = y_cols;
    auto output_size1 = x_interp_len;

    for (int i = 0; i < y_rows; i++) {
      int x0_i = 0;
      for (int j = 0; j < y_cols; j++) {
          y_grad_tensor[i * y_cols + j] = 0;
      }
      for (int j = 0; j < output_size1; j++) {
        auto xi = x_interp_tensor[j];
        // Zero outside interpolation range
        if ((xi < x_tensor[i * y_cols + 0]) || (xi > x_tensor[i * y_cols + len_x-1])) {
          continue;
        }
        while(x_tensor[i * y_cols + x0_i] < xi && x0_i < len_x - 1) { x0_i++; }

        for (int k = -order; k <= order; k++) {
          if (x0_i + k > len_x - 1) {
            break;
          }
          if (x0_i + k < 0) {
            continue;
          }
          auto d = 1.0f;
          if (x0_i + k != 0) {
            d = x_tensor[i * y_cols + x0_i + k] - x_tensor[i * y_cols +  x0_i + k - 1];
          } else {
            d = x_tensor[i * y_cols + 1] - x_tensor[i * y_cols + 0];
          }
          auto z = pi * (xi - x_tensor[i * y_cols + x0_i + k]) / d;
          auto kernel = 1.0f;
          if (z != 0.0f) {
            kernel = order * std::sin(z / order) * std::sin(z) / (z * z);
          }
          y_grad_tensor[i * y_cols + x0_i + k] += kernel * grad_tensor[i * x_interp_len + j];
        }
      }
    }
  }
};

template struct StoltGradFunctor<CPUDevice, float>;

} //namespace functor
} //namespace tensorflow

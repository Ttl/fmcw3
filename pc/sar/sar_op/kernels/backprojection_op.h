#ifndef TFBACKPROJECTION_H_
#define TFBACKPROJECTION_H_

#include <complex>
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device, typename Dtype>
struct BackprojectionFunctor {
  static void launch(const Device& d,
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
          float tx_offset);
};

template <typename Device, typename Dtype>
struct BackprojectionGradFunctor {
  static void launch(const Device& d,
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
          float tx_offset
          );
};

}  // namespace functor
}  // namespace tensorflow

#endif // TFBACKPROJECTION_H_

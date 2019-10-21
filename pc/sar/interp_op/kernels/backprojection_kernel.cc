#include "backprojection_op.h"
#include "tensorflow/core/framework/op.h"
#include <cmath>
#include <complex>

namespace tensorflow {
namespace functor {

template <typename Dtype>
struct BackprojectionFunctor<CPUDevice, Dtype> {
  static void launch(const CPUDevice& d,
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
          float tx_offset
          ) {
    constexpr auto pi = 3.14159265359f;
    constexpr auto c0 = 299792458.0f;
    constexpr auto pos_dim = 2;
    constexpr std::complex<float> j(0, 1);

    const auto ref_phase = j * 4.0f * pi * fc / c0;
    const auto dr = 1.0f / delta_r;

    // To radians and divide by 2 to get edge angle from the center.
    beamwidth = (pi / 180.0f) * (beamwidth / 2.0f);

    for (auto idx = 0; idx < Nx; idx++) {
      // X-coordinate.
      auto x = x0 + idx * dx;
      for (auto idy = 0; idy < Ny; idy++) {
        // Y-coordinate.
        auto y = y0 + idy * dy;
        img[idx * Ny + idy] = std::complex<float>(0, 0);
        for (auto i = 0; i < sweeps; i++) {
          // Sweep reference position.
          auto pos_x = pos[i * pos_dim + 0];
          auto pos_y = pos[i * pos_dim + 1];
          auto px = (x - pos_x);
          auto py = (y - pos_y);
          auto target_angle = std::atan2(px, py);
          if (std::abs(target_angle) > beamwidth) {
              // Pixel outside of the beam.
              continue;
          }
          // Calculate distance to the pixel.
          auto d = std::sqrt(px * px + py * py);
          if (tx_offset != 0.0f) {
            auto dtx = std::sqrt((px + tx_offset) * (px + tx_offset) + py * py);
            d = (d + dtx) / 2.0f;
          }

          std::complex<float> s;
          auto v_corr = -std::sin(target_angle) * v;
          auto sx = dr * d + v_corr;
          if (interp_order == 0) {
            // Nearest neighbor.
            auto id0 = static_cast<size_t>(std::round(sx));
            if (id0 < 0 || id0 >= sweep_samples) {
              continue;
            }
            s = data[i * sweep_samples + id0];
          } else {
             // Linear interpolation.
             auto id0 = static_cast<size_t>(std::floor(sx));
             auto id1 = id0 + 1;
             if (id0 < 0 || id1 >= sweep_samples) {
               continue;
             }
             auto s0 = data[i * sweep_samples + id0];
             auto s1 = data[i * sweep_samples + id1];

             s = s0 + (sx - id0) * (s1 - s0);
          }
          auto ref = std::exp(ref_phase * d);
          img[idx * Ny + idy] += s * ref;
        }
      }
    }
  };
};

template struct BackprojectionFunctor<CPUDevice, float>;

template <typename Dtype>
struct BackprojectionGradFunctor<CPUDevice, Dtype> {
  static void launch(const CPUDevice& d,
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
          ) {
    constexpr auto pi = 3.14159265359f;
    constexpr auto c0 = 299792458.0f;
    constexpr auto pos_dim = 2;
    constexpr std::complex<float> j(0, 1);

    const auto ref_phase = j * 4.0f * pi * fc / c0;
    const auto dr = 1.0f / delta_r;

    // To radians and divide by 2 to get edge angle from the center.
    beamwidth = (pi / 180.0f) * (beamwidth / 2.0f);

    for (auto i = 0; i < pos_dim * sweeps; i++) {
      pos_grad[i] = 0.0f;
    }

    for (auto idx = 0; idx < Nx; idx++) {
      // X-coordinate.
      auto x = x0 + idx * dx;
      for (auto idy = 0; idy < Ny; idy++) {
        // Y-coordinate.
        auto y = y0 + idy * dy;
        auto g = grad[idx * Ny + idy];
        for (auto i = 0; i < sweeps; i++) {
          // Sweep reference position.
          auto pos_x = pos[i * pos_dim + 0];
          auto pos_y = pos[i * pos_dim + 1];
          auto px = (x - pos_x);
          auto py = (y - pos_y);
          auto target_angle = std::atan2(px, py);
          if (std::abs(target_angle) > beamwidth) {
              // Pixel outside of the beam.
              continue;
          }
          // Calculate distance to the pixel.
          auto drx = std::sqrt(px * px + py * py);
          auto dtx = std::sqrt((px + tx_offset) * (px + tx_offset) + py * py);
          auto d = (drx + dtx) / 2.0f;

          // Linear interpolation.
          auto v_corr = -std::sin(target_angle) * v;
          auto sx = dr * d + v_corr;
          auto id0 = static_cast<size_t>(std::floor(sx));
          auto id1 = id0 + 1;
          if (id0 < 0 || id1 >= sweep_samples) {
            continue;
          }
          auto s0 = data[i * sweep_samples + id0];
          auto s1 = data[i * sweep_samples + id1];

          auto ds = s1 - s0;
          auto s = s0 + (sx - id0) * ds;

          auto ref = std::exp(ref_phase * d);
          auto dout = ref * (ref_phase * s + dr * ds);

          auto dx = 0.50f * (-px / drx - (px + tx_offset) / dtx);
          auto dy = 0.50f * (-py / drx - py / dtx);

          auto dout_dx = g * std::conj(dout * dx);
          auto dout_dy = g * std::conj(dout * dy);
          pos_grad[i * pos_dim + 0] += dout_dx.real();
          pos_grad[i * pos_dim + 1] += dout_dy.real();
        }
      }
    }
  };
};

template struct BackprojectionGradFunctor<CPUDevice, float>;

} //namespace functor
} //namespace tensorflow

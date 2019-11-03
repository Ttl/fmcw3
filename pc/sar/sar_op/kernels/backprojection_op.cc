
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "backprojection_op.h"

namespace tensorflow {

template <typename Device, typename Dtype>
class BackprojectionOp: public OpKernel {
public:

  explicit BackprojectionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("x0", &x0_));
    OP_REQUIRES_OK(context, context->GetAttr("dx", &dx_));
    OP_REQUIRES_OK(context, context->GetAttr("Nx", &Nx_));
    OP_REQUIRES_OK(context, context->GetAttr("y0", &y0_));
    OP_REQUIRES_OK(context, context->GetAttr("dy", &dy_));
    OP_REQUIRES_OK(context, context->GetAttr("Ny", &Ny_));
    OP_REQUIRES_OK(context, context->GetAttr("fc", &fc_));
    OP_REQUIRES_OK(context, context->GetAttr("v", &v_));
    OP_REQUIRES_OK(context, context->GetAttr("bw", &bw_));
    OP_REQUIRES_OK(context, context->GetAttr("tsweep", &tsweep_));
    OP_REQUIRES_OK(context, context->GetAttr("delta_r", &delta_r_));
    OP_REQUIRES_OK(context, context->GetAttr("interp_order", &interp_order_));
    OP_REQUIRES_OK(context, context->GetAttr("beamwidth", &beamwidth_));
    OP_REQUIRES_OK(context, context->GetAttr("tx_offset", &tx_offset_));

    OP_REQUIRES(context, Nx_ > 0,
                errors::InvalidArgument("Nx needs to be positive",
                                        Nx_));
    OP_REQUIRES(context, Ny_ > 0,
                errors::InvalidArgument("Ny needs to be positive",
                                        Ny_));

    OP_REQUIRES(context, delta_r_ > 0,
                errors::InvalidArgument("delta_r needs to be positive",
                                        delta_r_));

    OP_REQUIRES(context, interp_order_ >= 0,
                errors::InvalidArgument("interp_rder needs to be non-negative",
                                        interp_order_));
  }

  void Compute(OpKernelContext* context) override {

    const Tensor& pos = context->input(0);
    const Tensor& data = context->input(1);

    const TensorShape& pos_shape = pos.shape();
    const TensorShape& data_shape = data.shape();

    DCHECK_EQ(pos.dims(), 2);
    DCHECK_EQ(data.dims(), 2);

    DCHECK_EQ(pos.dim_size(1), 2);

    TensorShape img_shape;
    img_shape.AddDim(Nx_);
    img_shape.AddDim(Ny_);

    Tensor* img = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, img_shape, &img));

    // Velocity correction factor
    auto v = fc_ * v_ / ((bw_/tsweep_) * delta_r_) + tsweep_ * v_ / delta_r_;

    ::tensorflow::functor::BackprojectionFunctor<Device, Dtype>::launch(context->eigen_device<Device>(),
            pos.flat<float>().data(),
            data.flat<std::complex<float>>().data(),
            img->flat<std::complex<float>>().data(),
            static_cast<int>(data.shape().dim_size(0)),
            static_cast<int>(data.shape().dim_size(1)),
            x0_, dx_, Nx_,
            y0_, dy_, Ny_,
            fc_, v, delta_r_,
            interp_order_, beamwidth_, tx_offset_
            );

  }
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BackprojectionOp);
    float x0_;
    float dx_;
    int Nx_;
    float y0_;
    float dy_;
    int Ny_;
    float fc_;
    float v_;
    float bw_;
    float tsweep_;
    float delta_r_;
    int interp_order_;
    float beamwidth_;
    float tx_offset_;
};

template <typename Device, typename Dtype>
class BackprojectionGradOp: public OpKernel {
public:

  explicit BackprojectionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("x0", &x0_));
    OP_REQUIRES_OK(context, context->GetAttr("dx", &dx_));
    OP_REQUIRES_OK(context, context->GetAttr("Nx", &Nx_));
    OP_REQUIRES_OK(context, context->GetAttr("y0", &y0_));
    OP_REQUIRES_OK(context, context->GetAttr("dy", &dy_));
    OP_REQUIRES_OK(context, context->GetAttr("Ny", &Ny_));
    OP_REQUIRES_OK(context, context->GetAttr("fc", &fc_));
    OP_REQUIRES_OK(context, context->GetAttr("v", &v_));
    OP_REQUIRES_OK(context, context->GetAttr("bw", &bw_));
    OP_REQUIRES_OK(context, context->GetAttr("tsweep", &tsweep_));
    OP_REQUIRES_OK(context, context->GetAttr("delta_r", &delta_r_));
    OP_REQUIRES_OK(context, context->GetAttr("interp_order", &interp_order_));
    OP_REQUIRES_OK(context, context->GetAttr("beamwidth", &beamwidth_));
    OP_REQUIRES_OK(context, context->GetAttr("tx_offset", &tx_offset_));

    OP_REQUIRES(context, Nx_ > 0,
                errors::InvalidArgument("Nx needs to be positive",
                                        Nx_));
    OP_REQUIRES(context, Ny_ > 0,
                errors::InvalidArgument("Ny needs to be positive",
                                        Ny_));

    OP_REQUIRES(context, delta_r_ > 0,
                errors::InvalidArgument("delta_r needs to be positive",
                                        delta_r_));

    OP_REQUIRES(context, interp_order_ >= 0,
                errors::InvalidArgument("interp_rder needs to be non-negative",
                                        interp_order_));
  }

  void Compute(OpKernelContext* context) override {

    const Tensor& pos = context->input(0);
    const Tensor& data = context->input(1);
    const Tensor& grad = context->input(2);

    const TensorShape& pos_shape = pos.shape();
    const TensorShape& data_shape = data.shape();

    DCHECK_EQ(pos.dims(), 2);
    DCHECK_EQ(data.dims(), 2);
    DCHECK_EQ(grad.dims(), 2);

    DCHECK_EQ(pos.dim_size(1), 2);

    TensorShape pos_grad_shape;
    pos_grad_shape.AddDim(pos_shape.dim_size(0));
    pos_grad_shape.AddDim(pos_shape.dim_size(1));

    Tensor* pos_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, pos_grad_shape, &pos_grad));

    // Velocity correction factor
    auto v = fc_ * v_ / ((bw_/tsweep_) * delta_r_) + tsweep_ * v_ / delta_r_;

    ::tensorflow::functor::BackprojectionGradFunctor<Device, Dtype>::launch(context->eigen_device<Device>(),
            pos.flat<float>().data(),
            data.flat<std::complex<float>>().data(),
            grad.flat<std::complex<float>>().data(),
            pos_grad->flat<float>().data(),
            static_cast<int>(data.shape().dim_size(0)),
            static_cast<int>(data.shape().dim_size(1)),
            x0_, dx_, Nx_,
            y0_, dy_, Ny_,
            fc_, v, delta_r_,
            interp_order_, beamwidth_, tx_offset_
            );

  }
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BackprojectionGradOp);
    float x0_;
    float dx_;
    int Nx_;
    float y0_;
    float dy_;
    int Ny_;
    float fc_;
    float v_;
    float bw_;
    float tsweep_;
    float delta_r_;
    int interp_order_;
    float beamwidth_;
    float tx_offset_;
};

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("Backprojection").Device(DEVICE_CPU), BackprojectionOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("BackprojectionGrad").Device(DEVICE_CPU), BackprojectionGradOp<CPUDevice, float>);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("Backprojection").Device(DEVICE_GPU), BackprojectionOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("BackprojectionGrad").Device(DEVICE_GPU), BackprojectionGradOp<GPUDevice, float>);
#endif // GOOGLE_CUDA

}  // namespace tensorflow

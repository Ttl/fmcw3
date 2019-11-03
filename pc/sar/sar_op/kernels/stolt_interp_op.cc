
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "stolt_interp_op.h"

namespace tensorflow {

template <typename Device, typename Dtype>
class StoltInterpOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit StoltInterpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                       context->GetAttr("order", &order_));
    OP_REQUIRES(context, order_ > 0,
                errors::InvalidArgument("Order needs to be positive",
                                        order_));
  }

  /// \brief Interpolate
  /// \param context
  void Compute(OpKernelContext* context) override {

    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());

    // get the input tensors
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x_interp = context->input(2);

    // check shapes of input and weights
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_interp_shape = x_interp.shape();

    DCHECK_EQ(x.dims(), 2);
    DCHECK_EQ(x_interp.dims(), 1);
    DCHECK_EQ(y.dims(), 2);

    DCHECK_EQ(x_shape.dim_size(0), y_shape.dim_size(0));
    DCHECK_EQ(x_shape.dim_size(1), y_shape.dim_size(1));

    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(y_shape.dim_size(0));
    output_shape.AddDim(x_interp_shape.dim_size(0));

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    ::tensorflow::functor::StoltFunctor<Device, Dtype>::launch(context->eigen_device<Device>(),
            output->flat<std::complex<float>>().data(),
            y.flat<std::complex<float>>().data(),
            x.flat<float>().data(),
            x_interp.flat<float>().data(),
            static_cast<int>(y.shape().dim_size(0)),
            static_cast<int>(y.shape().dim_size(1)),
            static_cast<int>(x_interp.shape().dim_size(0)),
            order_);

  }
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(StoltInterpOp);
    int order_;
};

template <typename Device, typename Dtype>
class StoltInterpGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit StoltInterpGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                       context->GetAttr("order", &order_));
    OP_REQUIRES(context, order_ > 0,
                errors::InvalidArgument("Order needs to be positive",
                                        order_));
  }

  /// \brief Interpolate
  /// \param context
  void Compute(OpKernelContext* context) override {

    // some checks to be sure ...
    DCHECK_EQ(4, context->num_inputs());

    // get the input tensors
    const Tensor& x = context->input(0);
    const Tensor& y = context->input(1);
    const Tensor& x_interp = context->input(2);
    const Tensor& grad = context->input(3);

    // check shapes of input and weights
    const TensorShape& x_shape = x.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& x_interp_shape = x_interp.shape();

    DCHECK_EQ(x.dims(), 2);
    DCHECK_EQ(x_interp.dims(), 1);
    DCHECK_EQ(y.dims(), 2);

    DCHECK_EQ(x_shape.dim_size(0), y_shape.dim_size(0));
    DCHECK_EQ(x_shape.dim_size(1), y_shape.dim_size(1));

    // create output shape
    TensorShape y_grad_shape;
    y_grad_shape.AddDim(y_shape.dim_size(0));
    y_grad_shape.AddDim(y_shape.dim_size(1));

    // create output tensor
    Tensor* y_grad = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, y_grad_shape, &y_grad));

    ::tensorflow::functor::StoltGradFunctor<Device, Dtype>::launch(context->eigen_device<Device>(),
            y_grad->flat<std::complex<float>>().data(),
            y.flat<std::complex<float>>().data(),
            x.flat<float>().data(),
            x_interp.flat<float>().data(),
            grad.flat<std::complex<float>>().data(),
            static_cast<int>(y.shape().dim_size(0)),
            static_cast<int>(y.shape().dim_size(1)),
            static_cast<int>(x_interp.shape().dim_size(0)),
            order_);

  }
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(StoltInterpGradOp);
    int order_;
};

// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("StoltInterp").Device(DEVICE_CPU), StoltInterpOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("StoltInterpGrad").Device(DEVICE_CPU), StoltInterpGradOp<CPUDevice, float>);

#ifdef GOOGLE_CUDA
// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("StoltInterp").Device(DEVICE_GPU), StoltInterpOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("StoltInterpGrad").Device(DEVICE_GPU), StoltInterpGradOp<GPUDevice, float>);
#endif // GOOGLE_CUDA

}  // namespace tensorflow

/// \file stolt_interp.cc
/// \author Henrik Forsten
/// \brief Stolt interpolation for SAR image formation.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("StoltInterp")
  .Input("x: float32")
  .Input("y: complex64")
  .Input("x_interp: float32")
  .Output("y_interp: complex64")
  .Attr("order: int = 3")
  .SetShapeFn([](InferenceContext* c) {
    ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input));

    // y_interp.shape = [y.shape[0], x_interp.shape[0]]
    c->set_output(0, c->Matrix(c->Dim(c->input(1), 0), c->Dim(c->input(2), 0)));
    return ::tensorflow::Status::OK();
  })
  ;

REGISTER_OP("StoltInterpGrad")
  .Input("x: float32")
  .Input("y: complex64")
  .Input("x_interp: float32")
  .Input("grad: complex64")
  .Output("y_grad: complex64")
  .Attr("order: int = 3")
  .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input));

      // Gradients have the same shape as inputs.
      c->set_output(0, c->input(1));
      return ::tensorflow::Status::OK();
    })
  ;

} //namespace tensorflow

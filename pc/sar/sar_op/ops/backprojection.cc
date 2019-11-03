/// \file backprojection.cc
/// \author Henrik Forsten
/// \brief Back projection SAR image formation algorithm.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("Backprojection")
  .Input("pos: float32")
  .Input("data: complex64")
  .Output("img: complex64")
  .Attr("x0: float")
  .Attr("dx: float")
  .Attr("Nx: int")
  .Attr("y0: float")
  .Attr("dy: float")
  .Attr("Ny: int")
  .Attr("fc: float")
  .Attr("v: float")
  .Attr("bw: float")
  .Attr("tsweep: float")
  .Attr("delta_r: float")
  .Attr("interp_order: int = 1")
  .Attr("beamwidth: float = 90")
  .Attr("tx_offset: float = 0")
  .SetShapeFn([](InferenceContext* c) {
    ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));

    int Nx, Ny;
    c->GetAttr("Nx", &Nx);
    c->GetAttr("Ny", &Ny);

    c->set_output(0, c->Matrix(Nx, Ny));
    return ::tensorflow::Status::OK();
  })
  ;

REGISTER_OP("BackprojectionGrad")
  .Input("pos: float32")
  .Input("data: complex64")
  .Input("grad: complex64")
  .Output("pos_grad: float32")
  .Attr("x0: float")
  .Attr("dx: float")
  .Attr("Nx: int")
  .Attr("y0: float")
  .Attr("dy: float")
  .Attr("Ny: int")
  .Attr("fc: float")
  .Attr("v: float")
  .Attr("bw: float")
  .Attr("tsweep: float")
  .Attr("delta_r: float")
  .Attr("interp_order: int = 1")
  .Attr("beamwidth: float = 90")
  .Attr("tx_offset: float = 0")
  .SetShapeFn([](InferenceContext* c) {
    ShapeHandle input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &input));

    c->set_output(0, c->input(0));
    return ::tensorflow::Status::OK();
  })
  ;

} //namespace tensorflow

# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops

__all__ = []

def load_op(name, has_grad=False):
  """Load operation and add it to __all__ for imports.

  Args:
      name (str): name of operation without "_op" suffix
      has_grad (bool, optional): gradient (if exists) should be loaded as well

  Returns:
      functions
  """
  global __all__
  path = os.path.join(os.path.dirname(__file__), 'build/%s_op.so' % name)
  _module = tf.load_op_library(path)
  if has_grad:
    __all__.append('%s' % name)
    __all__.append('%s_grad' % name)
    return getattr(_module, '%s' % name), getattr(_module, '%s_grad' % name)
  else:
    __all__.append('%s' % name)
    return getattr(_module, '%s' % name)


stolt_interp, stolt_interp_grad = load_op('stolt_interp', has_grad=True)

backprojection, backprojection_grad = load_op('backprojection', has_grad=True)

@ops.RegisterGradient("StoltInterp")
def _StoltInterpGrad(op, *grads):
  x = op.inputs[0]
  y = op.inputs[1]
  x_interp = op.inputs[2]
  order = op.get_attr('order')
  grad = grads[0]
  y_grad = stolt_interp_grad(x, y, x_interp, grad, order)
  return (None, y_grad, None)

@ops.RegisterGradient("Backprojection")
def _BackprojectionGrad(op, *grads):
  pos = op.inputs[0]
  data = op.inputs[1]
  x0 = op.get_attr('x0')
  dx = op.get_attr('dx')
  Nx = op.get_attr('Nx')
  y0 = op.get_attr('y0')
  dy = op.get_attr('dy')
  Ny = op.get_attr('Ny')
  fc = op.get_attr('fc')
  v = op.get_attr('v')
  bw = op.get_attr('bw')
  tsweep = op.get_attr('tsweep')
  delta_r = op.get_attr('delta_r')
  interp_order = op.get_attr('interp_order')
  beamwidth = op.get_attr('beamwidth')
  tx_offset = op.get_attr('tx_offset')
  grad = grads[0]
  pos_grad = backprojection_grad(pos, data, grad, x0, dx, Nx,
                                 y0, dy, Ny, fc, v, bw, tsweep, delta_r,
                                 interp_order, beamwidth, tx_offset)
  return (pos_grad, None)

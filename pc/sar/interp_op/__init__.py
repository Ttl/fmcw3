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


@ops.RegisterGradient("StoltInterp")
def _StoltInterpGrad(op, *grads):
  x = op.inputs[0]
  y = op.inputs[1]
  x_interp = op.inputs[2]
  order = op.get_attr('order')
  grad = grads[0]
  y_grad = stolt_interp_grad(x, y, x_interp, grad, order)
  return (None, y_grad, None)

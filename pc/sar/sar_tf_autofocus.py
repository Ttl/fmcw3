#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.framework import ops
import uuid
import scipy.interpolate
from scipy.interpolate import interp1d
from interp_op import stolt_interp

from tensorflow.python.client import timeline

#@ops.RegisterGradient("StoltInterpolation")
#def _stolt_interpolation_grad_cc(op, grad):
#    return stolt_interp_grad_module.stolt_interpolation_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], order)
#
#tf.no_gradient("StoltInterpolationGrad")

#tf.no_gradient("StoltInterp")

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def r4_normalize(x, d, e=4):
    y = np.fft.rfft(x, axis=-1)
    n = d[-1]**e
    y = y*d**e/n
    return np.fft.irfft(y, axis=-1)

def rvp_compensation(x, f, kr):
    return x * np.exp(-1j * np.pi * f**2 / kr)

def hilbert_rvp(x, fs, kr):
    y = np.fft.fft(x, axis=-1)
    y[:,:y.shape[1]//2+1] = 0 # Zero positive frequencies
    # Residual video phase term compensation
    f = np.linspace(-fs/2, fs/2, y.shape[1])
    y *= np.exp(-1j * np.pi * f**2 / kr)
    return np.fft.ifft(y, axis=-1)

def tf_interp1d(x, y, x_interp, dtype=tf.float32):
    # Define the op in python
    def _interp1d(x, y, x_interp):
        out = np.zeros((x.shape[0], x_interp.shape[0]), dtype=y.dtype)
        for e, xi in enumerate(x):
            out[e,:] = interp1d(xi, y[e,:], kind='linear', bounds_error=False, fill_value=0)(x_interp)
        return out

    # Define the op's gradient in python
    def _interp1d_grad(x, x_interp, grad):
        grad_y = np.zeros_like(x, dtype=grad.dtype)

        for i in range(x.shape[0]):
            xi = x[i]
            x_indices = np.searchsorted(xi, x_interp)

            for e, idx in enumerate(x_indices):
                if idx >= len(xi):
                    break
                xv = x_interp[e]
                xf0 = xi[idx]
                if idx-1 >= 0:
                    d = xi[idx] - xi[idx-1]
                else:
                    d = xi[idx+1] - xi[idx]
                if idx-1 >= 0:
                    xf1 = xi[idx-1]
                else:
                    xf1 = xf0+d
                d0 = np.abs(xf0 - xv)
                d1 = np.abs(xf1 - xv)
                if idx == 0 and d1 > d:
                    continue
                if idx-1 >= 0:
                    grad_y[i, idx-1] += grad[i, e] * d0/d
                grad_y[i, idx] += grad[i, e] * d1/d

        return grad_y

    # An adapter that defines a gradient op compatible with TensorFlow
    def _interp1d_grad_op(op, grad):
        x = op.inputs[0]
        x_interp = op.inputs[2]
        y_grad = tf.py_func(_interp1d_grad, [x, x_interp, grad], dtype)
        #x_grad = tf.zeros(x.shape)
        #y_interp_grad = tf.zeros(x_interp.shape)
        return (None, y_grad, None)

    def _interp1d_hessian_op(op, grad):
        x = op.inputs[0]
        y_grad = tf.zeros(x.shape)
        return (None, y_grad, None)

    # Register the gradient with a unique id
    grad_name = "Interp1dGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_interp1d_grad_op)

    #hessian_name = "Interp1dGrad_" + str(uuid.uuid4())
    #tf.RegisterGradient(hessian_name)(_interp1d_hessian_op)

    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_interp1d, [x, y, x_interp], dtype)
    return output

channel = 1
rs = 0
speed = 1.5
max_range = 300
#FFT zero-padding amount, increases cross-range with reduced resolution
cross_range_padding = 2.5
#Dynamic range of final image in dB
dynamic_range = 90
# Interpolation grid spacing multiplier. 1 = No aliasing, >1 less points, aliases.
# 2 aliases fully to unused positive frequencies.
ky_delta_spacing = 1.5
interp_order = 5

window = np.hanning

c = 299792458

with open(sys.argv[1], 'rb') as f:
    settings, ch1, ch2 = pickle.load(f)

if channel == 1:
    data = np.array(ch1)
elif channel == 2:
    data = np.array(ch2)
else:
    raise ValueError("Invalid RX channel")

fs = settings['fs']
tsweep = settings['tsweep']
bw = settings['bw']
fc = settings['f0'] + bw/2
tdelay = settings['tdelay']
sweep_samples = len(data[0])
delta_crange = (tsweep + tdelay) * speed
print('Cross range delta {:.3f} m, {:.3f} lambda'.format(delta_crange, delta_crange/(c/fc)))

f = np.linspace(0, fs/2, sweep_samples//2+1)
d = f_to_d(f, bw, tsweep)

range0 = 0
range1 = c*(fs/2.0)*tsweep/(2*bw)
delta_range = range1/sweep_samples
crange0 = -len(data)*delta_crange/2.0
crange1 = len(data)*delta_crange/2.0
raw_extent = (range0, range1, crange0, crange1)

# Window data and hilbert transform
data = data * window(sweep_samples)
data = hilbert_rvp(data, fs, bw/tsweep)

phase = tf.get_variable("phase", data.shape[0], dtype=tf.float32,
                           initializer=tf.zeros_initializer())
                           #initializer=tf.truncated_normal_initializer())

# Zeropad cross-range
if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1)*data.shape[0])
    data = np.pad(data, ((zpad//2, zpad//2), (0, 0)), 'constant')
    phase_padded = tf.pad(phase, [[zpad//2, zpad//2]], 'constant')
else:
    phase_padded = phase

# Coordinates
kx = np.linspace(-np.pi/delta_crange, np.pi/delta_crange, len(data))
kr = np.linspace(((4*np.pi/c)*(fc - bw/2)), ((4*np.pi/c)*(fc + bw/2)), sweep_samples);
ky0 = (kr[0]**2 - kx[0]**2)**0.5
kr_delta = kr[1] - kr[0]
ky_delta = ky_delta_spacing * kr_delta
ky_even = np.arange(ky0, kr[-1], ky_delta)

print("Input size: {}".format(data.shape))
print("Output size: {}".format((data.shape[0], len(ky_even))))

t = tf.constant(np.linspace(0, tsweep, len(data[0])), dtype=tf.float32)
gamma = tf.constant(bw / tsweep, dtype=tf.float32)

# Apply phase correction
img = tf.constant(data, dtype=tf.complex64)
j = tf.constant(1j, dtype=tf.complex64)
#img = img * tf.exp(j * tf.cast(tf.expand_dims(phase_padded, -1), tf.complex64))
k = (4 * np.pi * fc / c)
img = img * tf.exp(j * tf.cast((1 + (gamma / (fc * k)) * tf.expand_dims(t, 0)) * tf.expand_dims(phase_padded, -1), tf.complex64))

# Along the track FFT
# TF calculates FFT always over the last axis. Tranpose data to take FFT over first axis,
img = tf.transpose(img)
img = tf.signal.fft(img)
img = tf.transpose(img)

# FFT shift
img1, img2 = tf.split(img, [img.shape[0].value - img.shape[0].value//2, img.shape[0].value//2], axis=0)
img = tf.concat([img2, img1], axis=0)

# Matched filter assumed to be identity (rs = 0).
assert rs == 0

#Stolt interpolation
ky_interp = tf.constant(ky_even, dtype=tf.float32)
kr_stack = np.tile(kr, (len(kx), 1))
ky = tf.constant((kr**2 - (kx**2)[:,np.newaxis])**0.5, dtype=tf.float32)
#img = tf_interp1d(ky, img, ky_interp, dtype=tf.complex64)
img = stolt_interp(ky, img, ky_interp, interp_order)
img = tf.signal.ifft2d(img)

abs_img = tf.abs(img)
abs_sum = tf.reduce_sum(abs_img)
abs_img = abs_img / abs_sum
entropy = -tf.reduce_sum(abs_img * tf.log(abs_img))

lr_ph = tf.placeholder(tf.float32)
opt_op = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(entropy)

init_op = tf.global_variables_initializer()

phase_grad = tf.gradients(entropy, phase)[0]
#phase_full_hessian = tf.hessians(entropy, phase)[0]
#phase_hessian = tf.linalg.tensor_diag_part(phase_full_hessian)
#abs_grad = tf.sqrt(tf.square(phase_grad) + tf.square(phase_hessian))
#update_step = -phase + tf.math.atan2(-phase_grad / abs_grad, -phase_hessian / abs_grad)
#update_step = -update_step + tf.constant(np.pi)
#update_op = tf.assign(phase, update_step)

#update_step = -phase_grad / phase_hessian
#update_op = tf.assign_add(phase, update_step)

loss = []
## Launch the graph in a session.
#with tf.Session() as sess:
#    sess.run(init_op)
#    h_np = np.zeros(phase.get_shape().as_list(), dtype=np.float32)
#    sess.run(tf.assign(phase, tf.constant(h_np)))
#    l0, hessian = sess.run([entropy, phase_hessian])
#    hx = 1e-3
#    h_np[100] = hx
#    h = tf.constant(h_np)
#    sess.run(tf.assign(phase, h))
#    lp = sess.run([entropy])
#    sess.run(tf.assign(phase, -h))
#    lm = sess.run([entropy])
#    hess = (lp - 2*l0 + lm)/hx**2
#    print(hess, hessian[100])

if 1:
    with tf.Session() as sess:
        sess.run(init_op)

        # Run the Op that initializes global variables.
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(phase_grad, options=options, run_metadata=run_metadata)

        loss = [0]

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline_01.json', 'w') as f:
            f.write(chrome_trace)

        exit(0)

if 1:
    with tf.Session() as sess:
        # Run the Op that initializes global variables.
        sess.run(init_op)
        for step in range(2):
            print('Step', step)
            lr = 0.1
            #if step > 80:
            #    lr = 0.01
            #l = sess.run([entropy], feed_dict={lr_ph:lr})
            l, _ = sess.run([entropy, opt_op], feed_dict={lr_ph:lr})
            #l, phase_grad, phase_hessian = sess.run([entropy, phase_grad_op, phase_hessian_op])
            #l = sess.run(entropy)
            #phase_grad = tf.gradients(entropy, phase)[0].eval()
            #phase_hessian = tf.hessians(entropy, phase)[0].eval()
            #phase_hessian = np.diag(phase_hessian)
            #print(phase.shape, phase_grad.shape, phase_hessian.shape)
            #abs_grad = np.sqrt(phase_grad**2 + phase_hessian**2)
            #a = phase_grad**2 + phase_hessian**2 / abs_grad
            #b = -phase.eval() + np.arctan2(-phase_grad / abs_grad, -phase_hessian / abs_grad)
            #print(b.shape)
            #c = phase_hessian + entropy
            #y = tf.constant(-b - np.pi)
            #tf.assign(phase, y)
            #l, _ = sess.run([entropy, opt_op], feed_dict={lr_ph: lr})
            loss.append(l)
        #phase_grad = phase_grad.eval()
        #phase_hessian = phase_hessian.eval()

        phases = phase.eval()
        st = img.eval()

print('Entropy', loss[-1])

#plt.figure()
#plt.plot(phase_grad, label='grad')
#plt.plot(phase_hessian, label='hessian')
#plt.legend(loc='best')

plt.figure()
plt.plot(loss)

plt.figure()
#plt.plot((180/np.pi)*phases)
plt.plot((180/np.pi)*np.unwrap(np.array(phases)))

with open('phase_correction.p', 'wb') as f:
    pickle.dump(phases, f)

plt.figure()
st = 20*np.log10(np.abs(st))
imgplot = plt.imshow(st, aspect='auto', interpolation='none', extent=[0, range1, crange0, crange1])
m = np.max(st)
#Limit the dynamic range to clean the rounding errors
imgplot.set_clim(m-dynamic_range,m)
plt.show()

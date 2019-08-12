#!/usr/bin/python3
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.client import timeline
from interp_op import stolt_interp
from taylor import taylor

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def hilbert_rvp(x, fs, gamma):
    y = np.fft.fft(x, axis=-1)
    y[:,:y.shape[1]//2+1] = 0 # Zero positive frequencies
    # Residual video phase term compensation
    f = np.linspace(-fs/2, fs/2, y.shape[1])
    y *= np.exp(1j * np.pi * f**2 / gamma)
    return np.fft.ifft(y, axis=-1)

channel = 1
rs = 0
v = 1.55
#FFT zero-padding amount, increases cross-range with reduced resolution
cross_range_padding = 2.0
#Dynamic range of final image in dB
dynamic_range = 80
# Interpolation grid spacing multiplier. 1 = No aliasing, >1 less points, aliases.
# 2 aliases fully to unused positive frequencies.
ky_delta_spacing = 1.6
interp_order = 3
transposed_image = False

#window = np.hanning
window = lambda N : taylor(N, nbar=4, level=-35)

c = 299792458

with open(sys.argv[1], 'rb') as f:
    settings, ch1, ch2 = pickle.load(f)

if channel == 1:
    data = np.array(ch1)
elif channel == 2:
    data = np.array(ch2)
else:
    raise ValueError("Invalid RX channel")

print(settings)
print(data.shape)
fs = settings['fs']
tsweep = settings['tsweep']
bw = settings['bw']
fc = settings['f0'] + bw/2
tdelay = settings['tdelay']
sweep_samples = len(data[0])
delta_x = (tsweep + tdelay) * v
print('Cross range delta {:.3f} m, {:.3f} lambda'.format(delta_x, delta_x/(c/fc)))

f = np.linspace(0, fs/2, sweep_samples//2+1)
d = f_to_d(f, bw, tsweep)

range0 = 0
range1 = c*(fs/2.0)*tsweep/(2*bw)
delta_range = range1/sweep_samples
crange0 = -len(data)*delta_x/2.0
crange1 = len(data)*delta_x/2.0
raw_extent = (range0, range1, crange0, crange1)

# Window data and hilbert transform
w = window(sweep_samples)
data = hilbert_rvp(w * data, fs, bw/tsweep)

phase = tf.get_variable("phase", data.shape[0], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

# Zeropad cross-range
if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1)*data.shape[0])
    data = np.pad(data, ((zpad//2, zpad//2), (0, 0)), 'constant')
    phase_padded = tf.pad(phase, [[zpad//2, zpad//2]], 'constant')
else:
    phase_padded = phase

# Coordinates
kx = np.linspace(-np.pi/delta_x, np.pi/delta_x, len(data))
dkr = np.linspace((4*np.pi/c)*(-bw/2), (4*np.pi/c)*(bw/2), sweep_samples)
kr = dkr + (4*np.pi/c)*fc
ky0 = (kr[0]**2 - kx[0]**2)**0.5
ky_delta = ky_delta_spacing * (kr[1] - kr[0])
ky_interp = np.arange(ky0, kr[-1], ky_delta)

print("Input size: {}".format(data.shape))
print("Output size: {}".format((data.shape[0], len(ky_interp))))

t = tf.constant(np.linspace(0, tsweep, len(data[0])), dtype=tf.float32)
gamma = tf.constant(bw / tsweep, dtype=tf.float32)

# Apply phase correction
img = tf.constant(data, dtype=tf.complex64)
j = tf.constant(1j, dtype=tf.complex64)
#img = img * tf.exp(j * tf.cast(tf.expand_dims(phase_padded, -1), tf.complex64))
wl = c / fc
rerr = -wl / (4*np.pi) * tf.expand_dims(phase_padded, -1)
img = img * tf.exp((-4 * np.pi * j / c) * tf.cast(rerr * (fc + gamma * tf.expand_dims(t, 0)), tf.complex64))

# Along the track FFT
# TF calculates FFT always over the last axis. Tranpose data to take FFT over first axis,
img = tf.transpose(img)
img = tf.signal.fft(img)
img = tf.transpose(img)

# FFT shift
img1, img2 = tf.split(img, [img.shape[0].value - img.shape[0].value//2, img.shape[0].value//2], axis=0)
img = tf.concat([img2, img1], axis=0)

ky = tf.constant((kr**2 - (kx**2)[:,np.newaxis])**0.5, dtype=tf.float32)

# Matched filtering
if rs != 0:
    mf = tf.exp(j * tf.cast(rs * ky, tf.complex64) + j * tf.cast(tf.expand_dims(dkr, 0) * tf.expand_dims(kx, -1), tf.complex64) * tf.cast(c * v / (4 * np.pi * gamma), tf.complex64))
else:
    mf = tf.exp(j * tf.cast(tf.expand_dims(dkr, 0) * tf.expand_dims(kx, -1), tf.complex64) * tf.cast(c * v / (4 * np.pi * gamma), tf.complex64))
img = img * mf

#Stolt interpolation
img = stolt_interp(ky, img, ky_interp, interp_order)

#img_shape = [data.shape[0], len(ky_interp)]
#w = np.outer(window(img_shape[0]), window(img_shape[1]))
#img = tf.signal.ifft2d(img * w)
img = tf.signal.ifft2d(img)

image_bins = int(img.shape[1].value / ky_delta_spacing)
img = img[:,:image_bins]
abs_img = tf.abs(img)
abs_img = abs_img / tf.reduce_sum(abs_img)
entropy = -tf.reduce_sum(abs_img * tf.log(abs_img))

phase_smoothness = tf.reduce_mean(tf.square(phase[1:] - phase[:-1]))

loss = entropy + 3 * phase_smoothness

lr_ph = tf.placeholder(tf.float32)
opt_op = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss)

init_op = tf.global_variables_initializer()

loss = []
with tf.Session() as sess:
    # Run the Op that initializes global variables.
    sess.run(init_op)
    for step in range(100):
        print('Step', step)
        lr = 0.1
        l, _ = sess.run([entropy, opt_op], feed_dict={lr_ph:lr})
        loss.append(l)

    phases = phase.eval()
    st = img.eval()

print('Entropy', loss[-1])

plt.figure()
plt.plot(loss)
plt.xlabel('Optimization steps')
plt.ylabel('Entropy')
plt.grid(True)

plt.figure()
plt.plot((180/np.pi)*phases)

with open('phase_correction.p', 'wb') as f:
    pickle.dump(phases, f)

plt.figure()
st = 20 * np.log10(np.abs(st))
aspect = (2 * cross_range_padding * crange1) / range1
if transposed_image:
    imgplot = plt.imshow(st.T, interpolation='none', origin='lower', aspect=1/aspect,
                     extent=[cross_range_padding * crange0, cross_range_padding * crange1, 0, range1, ])
else:
    imgplot = plt.imshow(st, interpolation='none', origin='lower', aspect=aspect,
                     extent=[0, range1, cross_range_padding * crange0, cross_range_padding * crange1])
plt.xlabel('Range [m]')
plt.ylabel('Cross-range [m]')
m = np.max(st)
imgplot.set_clim(m-dynamic_range,m)
plt.savefig('sar.png', dpi=600)
plt.show()

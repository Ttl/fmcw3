#!/usr/bin/python3

import tensorflow as tf
from sar_op import backprojection
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from scipy import interpolate as interp
from scipy.signal import hilbert
from math import ceil
from taylor import taylor

channel = 1
v0 = 1.55
#Dynamic range of final image in dB
dynamic_range = 80

azimuth_extent = [-160, 160]
range_extent = [0, 250]
antenna_aperture = 100e-3
antenna_beamwidth = 50 # degrees
# TX antenna position from RX antenna.
# First coordinate is azimuth, second range.
# Positive azimuth points to movement direction.
tx_position = [155e-3, 0]
upsample = 6

#window = lambda n: taylor(n, nbar=4, level=-35)
window = np.hamming

c = 299792458

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def rfft_rvp(x, fs, kr, upsample=1):
    y = np.fft.rfft(x, upsample*len(x[0]), axis=-1)
    # Residual video phase term compensation
    f = np.linspace(0, fs/2, y.shape[1])
    l = 1.0/len(y)
    y *= l * np.exp(-1j * np.pi * f**2 / kr)
    return np.conjugate(y)

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
gamma = bw / tsweep
fc = settings['f0'] + bw/2
tdelay = settings['tdelay']
sweeps = len(data)
sweep_samples = len(data[0])
delta_crange = (tsweep + tdelay) * v0
print('Cross range delta {:.3f} m, {:.3f} lambda'.format(delta_crange, delta_crange/(c/fc)))

t = np.linspace(0, tsweep, sweep_samples)
f = np.linspace(0, fs/2, sweep_samples//2+1)
sweep_d = f_to_d(f, bw, tsweep)
range0 = 0
range1 = c*(fs/2.0)*tsweep/(2*bw)
range_resolution = c / (2 * bw)

azimuth_resolution = antenna_aperture / 2
print('Range resolution {:.3f}, azimuth resolution {:.3f}'.format(range_resolution, azimuth_resolution))

azimuth_points = int(ceil((azimuth_extent[1] - azimuth_extent[0]) / azimuth_resolution))
x_axis = np.linspace(azimuth_extent[0], azimuth_extent[1], azimuth_points)

range_points = int(ceil((range_extent[1] - range_extent[0]) / range_resolution))
y_axis = np.linspace(range_extent[0], range_extent[1], range_points)

print('Image size: ({}, {})'.format(len(x_axis), len(y_axis)))
x, y = np.meshgrid(x_axis, y_axis, sparse=False, indexing='ij')

crange0 = -(len(data) - 1)*delta_crange/2.0
crange1 = (len(data) - 1)*delta_crange/2.0
raw_extent = (range0, range1, crange0, crange1)

data_x = np.linspace(crange0, crange1, len(data))

#Window data
data = data * window(sweep_samples)

print('Sweep points', sweep_samples)

#Hilbert transformation to get complex data
data = rfft_rvp(data, fs, bw/tsweep, upsample)

if 0:
    plt.plot(np.abs(data[0]))

if 0:
    shdata = 20*np.log10(np.abs(np.real([np.fft.rfft(r) for r in data])))
    plt.figure()
    plt.title('Raw data, range FFT')
    imgplot = plt.imshow(shdata, aspect='auto', interpolation='none', extent=raw_extent)
    plt.xlabel('Range [m]')
    plt.ylabel('Cross-range [m]')
    m = np.max(shdata)
    #Limit the dynamic range to clean the rounding errors
    imgplot.set_clim(m-dynamic_range,m)

if 0:
    plt.figure()
    plt.title('Raw data')
    plt.imshow(20*np.log10(np.abs(data)), aspect='auto', interpolation='none')
    plt.xlabel('Range wavenumber [1/m]')
    plt.ylabel('Cross-range [m]')

plt.show()

# Location of every pixel in the image
delta_rho = c / (2 * upsample * bw)

v = tf.Variable([[v0, 0] for i in range(len(data_x))], dtype=tf.float32)
# Enforce mean velocity to range direction to be zero to avoid geometric distortions.
vmeany = tf.expand_dims(tf.reduce_mean(v[:,1]), axis=-1)
vmeany = tf.pad(vmeany, [[1, 0]])
v = v - vmeany

pos = tf.math.cumsum((tsweep + tdelay) * v, axis=0)
pos = pos - tf.reduce_mean(pos, axis=0)
data = tf.constant(data, dtype=tf.complex64)

x0 = azimuth_extent[0]
Nx = azimuth_points
dx = (azimuth_extent[1] - azimuth_extent[0]) / Nx

y0 = range_extent[0]
Ny = range_points
dy = (range_extent[1] - range_extent[0]) / Ny

img = backprojection(pos, data, x0=x0, dx=dx, Nx=Nx, y0=y0, dy=dy, Ny=Ny, fc=fc, v=v0, bw=bw, tsweep=tsweep, delta_r=delta_rho, interp_order=1, beamwidth=antenna_beamwidth, tx_offset=tx_position[0])
abs_img = tf.abs(img)
abs_img = tf.clip_by_value(abs_img, 1e-10, 1e10)
abs_img = abs_img / tf.reduce_sum(abs_img)
entropy = -tf.reduce_sum(abs_img * tf.log(abs_img))

constant_v = tf.reduce_mean(v, axis=0)
constant_v = tf.reduce_mean(tf.square(v - constant_v), axis=0)

v_smoothness = tf.reduce_mean(tf.square(v[1:] - v[:-1]), axis=0)
loss = entropy + 5 * v_smoothness[0] + 2 * v_smoothness[1] + 1 * constant_v[0] + 1 * constant_v[1]

lr_ph = tf.placeholder(tf.float32)
opt_op = tf.train.MomentumOptimizer(learning_rate=lr_ph, momentum=0.9).minimize(loss)

init_op = tf.global_variables_initializer()

loss = []
with tf.Session() as sess:
    # Run the Op that initializes global variables.
    sess.run(init_op)
    for step in range(100):
        print('Step', step)
        lr = 1
        l, _ = sess.run([entropy, opt_op], feed_dict={lr_ph:lr})
        #l = sess.run([entropy], feed_dict={lr_ph:lr})
        loss.append(l)

    v, pos, img = sess.run([v, pos, img])

plt.figure()
plt.plot(loss)
plt.xlabel('Optimization steps')
plt.ylabel('Entropy')
plt.grid(True)

plt.figure()
plt.title('Optimized velocity')
plt.plot(v[:,0], label='Azimuth')
plt.plot(v[:,1], label='Range')
plt.legend(loc='best')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Sweep')
plt.grid(True)

print('Entropy', l)

plt.figure()
plt.title('Final image')

img += 1e-12

img = 20*np.log10(np.abs(img))
aspect = np.abs((azimuth_extent[1] - azimuth_extent[0]) / (range_extent[1] - range_extent[0]))
imgplot = plt.imshow(img, interpolation='none', extent=[range_extent[0], range_extent[1], azimuth_extent[0], azimuth_extent[1]], origin='lower')
m = np.max(img)
#Limit the dynamic range to clean the rounding errors
imgplot.set_clim(m-dynamic_range,m)
plt.show()

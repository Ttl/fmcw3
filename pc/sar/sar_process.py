#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from scipy import interpolate as interp
from scipy.signal import hilbert

channel = 1
rs = 0
speed = 1.55
#IFFT zero-padding amount, smooths final image
interpolate = 1
#FFT zero-padding amount, increases cross-range with reduced resolution
cross_range_padding = 2
#Dynamic range of final image in dB
dynamic_range = 80
ky_delta_spacing = 1.6

window = np.hanning

c = 299792458

def lanczos_interp1d(x, y):

    a = 3
    def finterp(xn):
        y_new = np.zeros(xn.shape[0], dtype=y.dtype)
        diff = np.ediff1d(x, to_end=x[-1]-x[-2])
        for e, xi in enumerate(xn):
            if xi < x[0] or xi > x[-1]:
                continue
            x0 = np.searchsorted(x, xi)
            for i in range(max(0, x0-a), min(len(x), x0+a+1)):
                z = (xi - x[i]) / diff[i]
                y_new[e] += y[i] * np.sinc(z) * np.sinc(z/a)
        return y_new

    return finterp

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

#Window data
data = data * window(sweep_samples)

print('Sweep points', sweep_samples)

#Hilbert transformation to get complex data
data = hilbert_rvp(data, fs, bw/tsweep)

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
    kr0 = (4*np.pi/c)*(fc - bw/2)
    kr1 = (4*np.pi/c)*(fc + bw/2)
    plt.imshow(data.real, aspect='auto', interpolation='none', extent=(kr0, kr1, crange0, crange1))
    plt.xlabel('Range wavenumber [1/m]')
    plt.ylabel('Cross-range [m]')

plt.show()

#Zeropad cross-range
if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1)*data.shape[0])
    data = np.pad(data, ((zpad//2, zpad//2), (0, 0)), 'constant')

kx = np.linspace(-np.pi/delta_crange, np.pi/delta_crange, len(data))
kr = np.linspace(((4*np.pi/c)*(fc - bw/2)), ((4*np.pi/c)*(fc + bw/2)), sweep_samples);

#along the track fft
cfft = np.fft.fftshift(np.fft.fft(data, axis=0), 0)

if 0:
    plt.figure()
    plt.title('Along track FFT phase')
    plt.imshow(np.angle(cfft), aspect='auto', extent=[kr[0], kr[-1], kx[0], kx[-1]])
    plt.figure()
    plt.title('Along track FFT magnitude')
    plt.imshow(np.abs(cfft), aspect='auto', extent=[kr[0], kr[-1], kx[0], kx[-1]])

#matched filter
if rs != 0:
    phi_mf = np.zeros(cfft.shape)
    for ii in range(cfft.shape[1]):
        for jj in range(cfft.shape[0]):
            phi_mf = rs*(kr[ii]**2-kx[jj]**2 )**0.5

    smf = np.exp(1j*phi_mf)
    cfft = cfft*smf

ky0 = (kr[0]**2 - kx[0]**2)**0.5
kr_delta = kr[1] - kr[0]
ky_delta = ky_delta_spacing * kr_delta
ky_even = np.arange(ky0, kr[-1], ky_delta)

st = np.zeros((cfft.shape[0], len(ky_even)), dtype=np.complex)

#Stolt interpolation
for i in range(len(kx)):
    ky = (kr**2 - kx[i]**2)**0.5
    #ci = interp.interp1d(ky, cfft[i], fill_value=0, bounds_error=False)
    ci = lanczos_interp1d(ky, cfft[i])
    st[i,:] = ci(ky_even)

if 0:
    plt.figure()
    plt.title('Stolt interpolation phase')
    plt.imshow(np.angle(st), aspect='auto', extent=[ky_even[0], ky_even[-1], kx[0], kx[-1]])
    plt.figure()
    plt.title('Stolt interpolation magnitude')
    plt.imshow(np.abs(st), aspect='auto', extent=[ky_even[0], ky_even[-1], kx[0], kx[-1]])

#Window again
#wx = window(st.shape[0])
#wy = window(st.shape[1])
#w = np.sqrt(np.outer(wx, wy))
#st = st * w

#IFFT to generate the image
st = np.fft.ifft2(st)

st_sum = np.sum(np.abs(st))
print('Entropy', -np.sum((np.abs(st)/st_sum) * np.log(np.abs(st)/st_sum)))

#Cross-range size of image in meters
crange = delta_crange*st.shape[0]/interpolate
max_range = range1 * ky_delta / (2 * kr_delta)

plt.figure()
plt.title('Final image')

st = 20*np.log10(np.abs(st))
imgplot = plt.imshow(st, aspect='auto', interpolation='none', extent=[0, range1,-crange/2.0,crange/2.0], origin='lower')
m = np.max(st)
#Limit the dynamic range to clean the rounding errors
imgplot.set_clim(m-dynamic_range,m)
plt.show()

#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import interpolate as interp

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def sinc_interp1d(x, y):

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

dynamic_range = 90

with open('sar_image.p', 'rb') as f:
    settings, st, kx, ky_even, (range0, range1, crange0, crange1) = pickle.load(f)

if 1:
    #st = st[3400:4150,443:518]
    (x0, y0, x1, y1) = (700, 2400, 750, 3100)
    #(x0, y0, x1, y1) = (280, 4600, 320, 5100)
    if 0:
        original_size = True
        crop = st
        crop[:y0,:] = 0
        crop[y1:,:] = 0
        crop[:,:x0] = 0
        crop[:,x1:] = 0
    else:
        original_size = False
        crop = st[y0:y1,x0:x1]

if 0:
    x = np.abs(np.fft.fft(crop, axis=0))

    if 0:
        plt.figure()
        plt.plot(np.abs(st[int(0.25*len(x)),:]))
        plt.plot(np.abs(st[int(0.5*len(x)),:]))
        plt.plot(np.abs(st[int(0.75*len(x)),:]))

    ma_len = 15
    argmax = np.argmax(x, axis=1)
    absmax = np.amax(x, axis=1)
    argmax = moving_average(argmax, ma_len)
    absmax = moving_average(absmax, ma_len)

    valid = absmax > max(absmax)/20.

    plt.figure()
    plt.plot(valid * argmax)
    #plt.plot(valid * absmax)

if 1:
    crop_fft2 = np.fft.fft2(crop)
    im = np.zeros(crop.shape, dtype=np.complex)
    ky = np.linspace(ky_even[0], ky_even[-1], st.shape[1])
    kx = np.linspace(kx[0], kx[-1], st.shape[0])
    if not original_size:
        print(len(kx), len(ky))
        ky = ky[x0:x1]
        kx = kx[y0:y1]
        print(len(kx), len(ky))

    #Stolt interpolation
    for i in range(len(kx)):
        #if original_size and i < y0 or i > y1:
        #    continue
        kr = (kx[i]**2 + ky**2)**0.5
        print(kr[0], kr[-1], ky[0], ky[-1])
        #ci = interp.interp1d(ky, crop_fft2[i], fill_value=0, bounds_error=False)
        ci = sinc_interp1d(ky, crop_fft2[i])
        im[i,:] = ci(kr)

    im = np.fft.ifft(np.fft.ifftshift(im, 0), axis=0)
    #im = np.fft.rfft(np.real(im), axis=1)
    im = np.fft.fft(im, axis=1)
    plt.figure()
    plt.imshow(20*np.log10(np.abs(im)), aspect='auto', interpolation='none')

if 0:
    print(st.shape)

    plt.figure()
    stm = 20*np.log10(np.abs(st))
    imgplot = plt.imshow(stm, aspect='auto', interpolation='none')#, extent=[0, range1,crange0, crange1])
    m = np.max(stm)
    #Limit the dynamic range to clean the rounding errors
    imgplot.set_clim(m-dynamic_range,m)

    plt.figure()
    stp = np.angle(st)
    imgplot = plt.imshow(stp, aspect='auto', interpolation='none')#, extent=[0, range1,crange0, crange1])

plt.show()

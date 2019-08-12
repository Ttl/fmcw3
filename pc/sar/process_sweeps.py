#!/usr/bin/python3
import sys
import numpy as np
import ast
import pickle

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def r4_normalize(x, d, e=2):
    n = d[-1]**e
    return x*d**e/n

def read_settings(f):
    f.seek(0)
    ls = f.read(2)
    if len(ls) != 2:
        raise ValueError('Too short read when reading settings')
    #ls = list(map(ord, ls))
    ls = ls[0] + (ls[1] << 8)
    settings = f.read(ls)
    if len(settings) != ls:
        raise ValueError('Too short read when reading settings')
    header = settings[:6]
    if header.decode('utf-8') != 'fmcw3;':
        raise ValueError('Missing header')
    settings = ast.literal_eval(settings[7:].decode('utf-8'))
    return settings

start = b'\x7f'
fs = 2e6
c = 299792458.0
fir_gain = 9.0
adc_ref = 1
adc_bits = 12

sweeps_to_read = None
decimate_sweeps = 2
fix_saturation = False

l = []
ch1 = []
ch2 = []

def find_start(f):
    done = False
    while not done:
        r = f.read(1)
        if len(r) == b'':
            return
        if r != start:
            continue
        n = ord(f.read(1))
        done = True
        pos = f.tell()
        for j in range(2):
            f.read(channels*samples_in_sweep*2)
            if f.read(1) != start:
                done = False
            x = f.read(1)
            if len(x) == 0:
                return
            if ord(x) != n+1+j:
                done = False
        f.seek(pos)
    return

tstart = float(sys.argv[2])
tend = float(sys.argv[3])

if tstart > tend:
    raise ValueError()

with open(sys.argv[1], 'rb') as f:
    settings = read_settings(f)
    f0 = settings['f0']
    bw = settings['bw']
    tsweep = settings['tsweep']
    tdelay = settings['tdelay']
    half = settings['downsampler']
    quarter = settings['quarter']
    decimate = settings['decimate']
    channels = int(settings['a']) + int(settings['b'])

    if half:
        fs /= 2
        if quarter:
            fs /= 2

    fc = f0+bw/2
    wl = c/(f0+bw/2)
    samples_in_sweep = int(tsweep*fs)

    if sweeps_to_read != None:
        samples = (2+2*samples_in_sweep*channels)*sweeps_to_read
    else:
        samples = None

    find_start(f)
    print('start', f.tell())
    i = 0
    j = 0
    sweep_count = 0
    previous_n = None
    while samples == None or i < samples:
        n = f.read(2)
        j += 2
        i += 2
        if len(n) != 2:
            break
        w = n[0] + (n[1] << 8)
        w = twos_comp(w, 16)
        l.append(w)
        if j == channels*samples_in_sweep*2:
            s, n = f.read(2)
            restart = False
            if s != start[0]:
                print('Lost track of start at {}'.format(f.tell()))
                restart = True
            if restart == False and previous_n != None:
                if n != (previous_n+1)&0xff:
                    print('Lost a sweep. Previous {}, now {} at {}'.format(previous_n, n, f.tell()))
                    restart = True
            if restart:
                find_start(f)
                print('Jumped to {}'.format(f.tell()))
                previous_n = None
                l = []
                j = 0
                continue
            previous_n = n
            j = 0
            if decimate_sweeps <= 1 or sweep_count >= decimate_sweeps:
                if channels == 2:
                    ch1.append(l[::2])
                    ch2.append(l[1::2])
                else:
                    ch1.append(l)
                l = []
                sweep_count = 1
            else:
                l = []
                sweep_count += 1

settings['fs'] = fs
print('Done reading')

ch1 = np.array(ch1)
ch2 = np.array(ch2)
decimate_sweeps *= decimate

settings['tdelay'] = (tsweep+tdelay)*decimate_sweeps - tsweep
print('Delay', settings['tdelay'])
lines = max(len(ch1), len(ch2))
print('Lines', lines)

tslow = np.linspace(0, decimate_sweeps*(lines-1)*(tsweep+tdelay), lines)
f = np.linspace(0, fs/2, len(ch1[0])//2+1)
d = f_to_d(f, bw, tsweep)

nstart = np.searchsorted(tslow, tstart)
nend = np.searchsorted(tslow, tend)

print('nstart {}, nend {}, length {}'.format(nstart, nend, max(len(ch1), len(ch2))))

if len(ch1) > 0:
    ch1 = ch1[nstart:nend]

if len(ch2) > 0:
    ch2 = ch2[nstart:nend]

m = 0.0
if fix_saturation:
    for ch in [ch1, ch2]:
        if len(ch) == 0:
            continue
        for e, s in enumerate(ch):
            fs = np.fft.rfft(s)
            y = np.abs(fs)
            y /= len(y)*(fir_gain*2**(adc_bits-1))
            if np.max(y) > 1.02:
                n = np.argmax(y)
                fs[int(1.5*n):] *= 0.2
                x = np.fft.irfft(fs)
                print('Saturation', e)
                ch[e] = x

with open('sweeps.p', 'wb') as f:
    pickle.dump((settings, ch1, ch2), f)

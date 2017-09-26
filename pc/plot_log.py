import sys
import matplotlib.pyplot as plt
import numpy as np
import ast
from scipy.signal import butter, filtfilt

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val                         # return positive value as is

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def r4_normalize(x, d, e=1.5):
    n = d[-1]**e
    return x*d**e/n

def read_settings(f):
    f.seek(0)
    ls = f.read(2)
    if len(ls) != 2:
        raise ValueError('Too short read when reading settings')
    ls = map(ord, ls)
    ls = ls[0] + (ls[1] << 8)
    settings = f.read(ls)
    if len(settings) != ls:
        raise ValueError('Too short read when reading settings')
    header = settings[:6]
    if header != 'fmcw3;':
        raise ValueError('Missing header')
    settings = ast.literal_eval(settings[7:])
    return settings

start = '\x7f'
fs = 2e6
fir_gain = 9.0
c = 299792458.0
adc_ref = 1
adc_bits = 12

max_range = 50
d_antenna = 28e-3 #Antenna distance
#channel_dl = 0e-3 #Channel length difference
angle_limit = 55
channel_offset = 21
swap_chs = True
sweeps_to_read = None
angle_pad = 100
decimate_sweeps = 1
kaiser_beta = 6

l = []
ch1 = []
ch2 = []

def find_start(f):
    done = False
    while not done:
        r = f.read(1)
        if r == '':
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

with open(sys.argv[1], 'r') as f:
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
    print 'start', f.tell()
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
        w = map(ord, n)
        w = w[0] + (w[1] << 8)
        w = twos_comp(w, 16)
        l.append(w)
        if j == channels*samples_in_sweep*2:
            s, n = f.read(2)
            restart = False
            if s != start:
                print 'Lost track of start at {}'.format(f.tell())
                restart = True
            n = ord(n)
            if restart == False and previous_n != None:
                if n != (previous_n+1)&0xff:
                    print 'Lost a sweep. Previous {}, now {} at {}'.format(previous_n, n, f.tell())
                    restart = True
            if restart:
                find_start(f)
                print 'Jumped to {}'.format(f.tell())
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
                sweep_count = 0
            else:
                l = []
                sweep_count += 1

print 'Done reading'

ch1 = np.array(ch1)
ch2 = np.array(ch2)
decimate_sweeps *= decimate

print 'Delay', (tsweep+tdelay)*decimate_sweeps

t = np.linspace(0, tsweep, len(ch1[0]))
f = np.linspace(0, fs/2, len(ch1[0])//2+1)
d = f_to_d(f, bw, tsweep)
angles = 180/np.pi*np.arcsin(np.linspace(1, -1, angle_pad)*wl/(2*d_antenna))
angle_mask = ~(np.isnan(angles) + (np.abs(angles) > angle_limit))
angles_masked = angles[angle_mask]
plot_i = -12

if 0:
    #Add 0 dBFs signal for testing purposes
    ch1[plot_i] += fir_gain*2**(adc_bits-1)*np.sin(2*np.pi*50e3*np.linspace(0, tsweep, len(ch1[0])))

w = np.kaiser(len(ch1[0]), kaiser_beta)
w *= len(w)/np.sum(w)

if 1:
    subtract_background = False
    subtract_clutter = False

    if subtract_background:
        background1 = np.zeros(len(ch1[0]))
        background2 = np.zeros(len(ch1[0]))
        for i in xrange(len(ch1[0])):
            x1 = 0
            x2 = 0
            for j in xrange(len(ch1)):
                x1 += ch1[j][i]
                x2 += ch2[j][i]
            background1[i] = (x1/len(ch1))
            background2[i] = (x2/len(ch1))

    angle_window = np.kaiser(len(angles), 150)
    clim = None

    if 0:
        coefs = [0.008161818583356717,
        -0.34386493885120994,
        0.65613506114879,
        -0.34386493885120994,
        0.008161818583356717]

        moving_average = len(coefs)
    else:
        coefs = [1]
        moving_average = 1

    for plot_j in range(moving_average-1, len(ch1)):
        fxm = None
        for k in range(moving_average):
            plot_i = plot_j-moving_average+k
            if subtract_background:
                a = w*(ch1[plot_i] - background1)
                b = w*(ch2[plot_i] - background2)
            elif subtract_clutter:
                if plot_i == 0:
                    continue
                a = w*(ch1[plot_i] - ch1[plot_i-1])
                b = w*(ch2[plot_i] - ch2[plot_i-1])
            else:
                a = w*ch1[plot_i]
                b = w*ch2[plot_i]
            a = np.fft.rfft(a)
            b = np.fft.rfft(b)

            #b *= np.exp(-1j*2*np.pi*channel_dl/(c/(f0+bw/2)))
            b *= np.exp(-1j*2*np.pi*channel_offset*np.pi/180)
            if swap_chs:
                x = np.concatenate((b, a)).reshape(2, -1)
            else:
                x = np.concatenate((a, b)).reshape(2, -1)
            fx = np.fft.fftshift(np.fft.fft(x, axis=0, n=angle_pad), axes=0)

            fx = r4_normalize(fx, d)

            if clim == None:
                max_range_i = np.searchsorted(d, max_range)
                clim = np.max(20*np.log10(np.abs(fx[:max_range_i,:]))) + 10

            if fxm == None:
                fxm = coefs[0]*fx
            else:
                fxm += coefs[k]*fx

        fx = fxm
        if 1:
            for j in range(fx.shape[1]):
                fj = fx[:,j]
                m = np.argmax(np.abs(fj))
                window = np.roll(angle_window, int(round(-fx.shape[0]/2 - m)))
                fx[:,j] *= window

        fx = fx[angle_mask]
        fxdb = 20*np.log10(np.abs(fx))

        fig = plt.figure()
        if 0:
            ax = fig.add_subplot(111, polar=True)
            imgplot = ax.pcolormesh(angles_masked*np.pi/180, d, fxdb.transpose())
        elif 0:
            r, t = np.meshgrid(d, angles_masked*np.pi/180)
            x = r*np.cos(t)
            y = -r*np.sin(t)
            imgplot = plt.pcolormesh(x, y, fxdb)
            plt.colorbar()
            ylim = 90*np.sin(angles_masked[0]*np.pi/180)
            #plt.ylim([-ylim, ylim])
            plt.ylim([-30, 30])
            plt.xlim([d[0], max_range])
            plt.ylabel("Cross-range [m]")
        else:
            imgplot = plt.pcolormesh(d, angles_masked, fxdb)
            plt.colorbar()
            plt.ylim([angles_masked[0], angles_masked[-1]])
            plt.xlim([d[0], max_range])
            plt.ylabel("Angle [$^o$]")
        imgplot.set_clim(clim-50,clim)
        plt.title('{0:.2f} s'.format( plot_i*(tsweep+tdelay)*decimate_sweeps))
        plt.xlabel("Range [m]")
        plt.savefig('img/range_{:04d}.png'.format(plot_i))
        plt.close()


if 0:
    plt.figure()
    plt.plot(t, np.array(ch1[plot_i])/(fir_gain*2**(adc_bits-1)))
    if channels == 2:
        plt.plot(t, np.array(ch2[plot_i])/(fir_gain*2**(adc_bits-1)))
    plt.title('IF time-domain waveforms')
    plt.ylabel("Amplitude [V]")
    plt.xlabel("Time [s]")

if 0:
    x1 = np.array(ch1[plot_i], dtype=np.float)
    x2 = np.array(ch2[plot_i], dtype=np.float)

    x1 *= w/(fir_gain*2**(adc_bits-1))
    x2 *= w/(fir_gain*2**(adc_bits-1))
    fx1 = 2*np.fft.rfft(x1)/(len(x1))
    fx2 = 2*np.fft.rfft(x2)/(len(x2))
    #fx1 = r4_normalize(fx1, d)
    fx1 = 20*np.log10(np.abs(fx1))
    fx2 = 20*np.log10(np.abs(fx2))

    print np.mean(fx1[40:])
    print np.mean(fx2[40:])
    plt.figure()
    plt.plot(d, fx1, label='Channel 1')
    plt.plot(d, fx2, label='Channel 2')
    plt.legend(loc='best')
    plt.title('IF spectrum')
    plt.ylabel("Amplitude [dBFs]")
    plt.xlabel("Distance [m]")


if 1:
    sweeps = ch2

    subtract_clutter = True
    subtract_background = False

    sw_len = len(ch1[0])

    if subtract_background:
        background = []
        for i in xrange(sw_len):
            x = 0
            for j in xrange(len(sweeps)):
                x += sweeps[j][i]
            background.append(x/len(sweeps))

    lines = len(sweeps)
    print lines, "lines"
    fourier_len = len(sweeps[0])/2
    max_range_index = int((4*bw*fourier_len*max_range)/(c*fs*tsweep))
    max_range_index = min(max_range_index, sw_len//2)
    print max_range_index
    im = np.zeros((max_range_index-2, lines))
    w = np.kaiser(sw_len, kaiser_beta)
    m = 0

    for e in xrange(len(sweeps)):
        sw = sweeps[e]
        if subtract_clutter and e > 0:
            sw = [sw[i] - sweeps[e-1][i] for i in xrange(sw_len)]
        if subtract_background:
            sw = [sw[i] - background[i] for i in xrange(sw_len)]
        sw = [sw[i]*w[i] for i in xrange(len(w))]
        fy = np.fft.rfft(sw)[3:max_range_index+1]
        fy = 20*np.log10((adc_ref/(2**(adc_bits-1)*fir_gain*max_range_index))*np.abs(fy))
        fy = np.clip(fy, -100, float('inf'))
        m = max(m,max(fy))
        im[:,e] = np.array(fy)

    if 1:
        f = fs/2.0

        t = np.linspace(0, decimate_sweeps*lines*(tsweep+tdelay), im.shape[1])
        xx, yy = np.meshgrid(
            #np.linspace(0,im.shape[1]-1, im.shape[1]),
            t,
            np.linspace(0, c*max_range_index*fs/(2*sw_len)/((bw/tsweep)), im.shape[0]))
        plt.figure()
        plt.ylabel("Range [m]")
        plt.xlabel("Time [s]")
        plt.xlim([t[0], t[-1]])
        plt.title(' Range-time plot')
        imgplot = plt.pcolormesh(xx,yy,im)
        imgplot.set_clim(m-80,m)
        plt.colorbar()
        #Save png of the plot
        #image.imsave('range_time_raw.png', np.flipud(im))
        #plt.savefig('range_time.png', dpi=500)

plt.show()

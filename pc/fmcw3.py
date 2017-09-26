import pylibftdi as ftdi
from adf4158 import ADF4158
from queue import Queue, Empty
from threading import Thread
import datetime

class FMCW3():
    def __init__(self):
        SYNCFF = 0x40
        SIO_RTS_CTS_HS = (0x1 << 8)
        self.device = ftdi.Device(mode='b', interface_select=ftdi.INTERFACE_A)
        self.device.open()
        self.device.ftdi_fn.ftdi_set_bitmode(0xff, SYNCFF)
        self.device.ftdi_fn.ftdi_read_data_set_chunksize(0x10000)
        self.device.ftdi_fn.ftdi_write_data_set_chunksize(0x10000)
        self.device.ftdi_fn.ftdi_setflowctrl(SIO_RTS_CTS_HS)
        self.device.flush()

        self.pll = ADF4158()
        self.fclk = 40e6
        self.fpd_freq = self.fclk/2

    def close(self):
        self.device.close()
        return

    def send_packet(self, x, cmd):
        try:
            l = len(x)
        except TypeError:
            l = 1
            x = [x]
        header = [0xaa, l, cmd]
        b = bytearray(header+x)
        #print map(hex, map(ord, str(b)))
        return self.device.write(str(b))

    def clear_gpio(self, led=False, pa_off=False, mix_enbl=False, adf_ce=False):
        w = 0
        if led:
            w |= 1 << 0
        if pa_off:
            w |= 1 << 1
        if mix_enbl:
            w |= 1 << 2
        if adf_ce:
            w |= 1 << 3
        return self.send_packet(w, 0)

    def set_gpio(self, led=False, pa_off=False, mix_enbl=False, adf_ce=False):
        w = 0
        if led:
            w |= 1 << 0
        if pa_off:
            w |= 1 << 1
        if mix_enbl:
            w |= 1 << 2
        if adf_ce:
            w |= 1 << 3
        return self.send_packet(w, 1)

    def clear_adc(self, oe1=False, oe2=False, shdn1=False, shdn2=False):
        w = 0
        if oe1:
            w |= 1 << 0
        if oe2:
            w |= 1 << 1
        if shdn1:
            w |= 1 << 2
        if shdn2:
            w |= 1 << 3
        return self.send_packet(w, 2)

    def set_adc(self, oe1=False, oe2=False, shdn1=False, shdn2=False):
        w = 0
        if oe1:
            w |= 1 << 0
        if oe2:
            w |= 1 << 1
        if shdn1:
            w |= 1 << 2
        if shdn2:
            w |= 1 << 3
        return self.send_packet(w, 3)

    def write_pll_reg(self, n):
        reg = self.pll.registers[n]
        w = [(reg & 0xff) | n, (reg >> 8) & 0xff, (reg >> 16) & 0xff, (reg >> 24) & 0xff]
        return self.send_packet(w, 4)

    def write_pll(self):
        self.write_pll_reg(7)
        self.pll.write_value(step_sel=0)
        self.write_pll_reg(6)
        self.pll.write_value(step_sel=1)
        self.write_pll_reg(6)
        self.pll.write_value(dev_sel=0)
        self.write_pll_reg(5)
        self.pll.write_value(dev_sel=1)
        self.write_pll_reg(5)
        for i in range(4,-1,-1):
            self.write_pll_reg(i)

    def set_sweep(self, fstart, bw, length, delay):
        real_delay = self.pll.freq_to_regs(fstart, self.fpd_freq, bw, length, delay)
        self.write_pll()
        return real_delay

    def write_sweep_timer(self, length):
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 5)

    def write_sweep_delay(self, length):
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 6)

    def set_channels(self, a=True, b=True):
        w = 0
        if a:
            w |= 1 << 0
        if b:
            w |= 1 << 1
        return self.send_packet(w, 7)

    def set_downsampler(self, enable=True, quarter=False):
        w = 0
        if enable:
            w |= 1 << 0
        if quarter:
            w |= 1 << 1
        return self.send_packet(w, 8)

    def write_decimate(self, decimate):
        if decimate > 2**16-1 or decimate < 1:
            raise ValueError("Invalid decimate value")
        decimate = int(decimate) - 1
        w = [decimate & 0xff, (decimate >> 8) & 0xff]
        return self.send_packet(w, 9)

    def write_pa_off_timer(self, length):
        length = int(self.fclk*length)
        w = [(length & 0xff), (length >> 8) & 0xff, (length >> 16) & 0xff, (length >> 24) & 0xff]
        return self.send_packet(w, 10)

    def clear_buffer(self):
        return self.send_packet(0, 11)

class Writer(Thread):
    def __init__(self, filename, queue):
        Thread.__init__(self)
        self.queue = queue
        self.f = open(filename, 'wb')

    def run(self):
        wrote = 0
        while True:
            try:
                d = self.queue.get(True, 0.5)
            except Empty:
                print 'Timeout', wrote
                self.f.close()
                return
            if len(d) == 0:
                print 'Closed', wrote
                self.f.close()
                return
            else:
                self.f.write(d)
                wrote += len(d)

fmcw3 = FMCW3()

f0 = 5.3e9
bw = 600e6
tsweep = 1e-3
tdelay = 2e-3
pa_off_advance = 0.2e-3
decimate = 3
ch_a = True
ch_b = True
downsampler = True
quarter = False

fmcw3.set_gpio(led=True, adf_ce=True)
fmcw3.set_adc(oe2=True)
fmcw3.clear_adc(oe1=True, shdn1=True, shdn2= True)
delay = fmcw3.set_sweep(f0, bw, tsweep, tdelay)

fmcw3.set_downsampler(enable=downsampler, quarter=quarter)
fmcw3.write_sweep_timer(tsweep)
fmcw3.write_sweep_delay(tdelay)
fmcw3.write_decimate(decimate)
fmcw3.write_pa_off_timer(tdelay - pa_off_advance)
fmcw3.clear_gpio(pa_off=True)
fmcw3.clear_buffer()

q = Queue()

date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

settings_dict = {'date': date, 'f0':f0, 'bw': bw, 'tsweep': tsweep, 'tdelay': tdelay, 'a': ch_a, 'b': ch_b, 'decimate': decimate, 'downsampler': downsampler, 'quarter': quarter}

settings = 'fmcw3; {}'.format(settings_dict)

ls = len(settings)
len_settings = ''.join(map(chr, [ls & 0xff, (ls >> 8) & 0xff]))

settings = len_settings+settings

q.put(settings)

writer = Writer('fmcw3.log', q)
writer.start()

fmcw3.set_channels(a=ch_a, b=ch_b)

try:
    while True:
        r = fmcw3.device.read(0x10000)
        if len(r) != 0:
            q.put(r)
finally:
    fmcw3.set_adc(oe1=True, shdn1=True, shdn2= True)
    fmcw3.set_channels(a=False, b=False)
    fmcw3.clear_gpio(led=True, adf_ce=True)
    fmcw3.set_gpio(pa_off=True)
    fmcw3.clear_buffer()
    fmcw3.close()
    print 'Done'
    q.put('')
    writer.join()

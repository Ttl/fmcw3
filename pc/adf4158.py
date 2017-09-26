from __future__ import division
from fractions import Fraction
from math import ceil, log

class ADF4158():
    def __init__(self):
        #Register definitions:
        # (Register: name: (First bit, Length, [value]))
        self.register_def = {
                0:{'ramp_on':(31, 1), 'muxout':(27, 4), 'n':(15,12), 'frac_msb':(3, 12)},
                1:{'reserved1':(28,4), 'frac_lsb':(15, 12), 'reserved2':(3, 12)},
                2:{'reserved3':(29, 3), 'csr_en':(28, 1), 'cp_current':(24, 4), 'reserved4':(23, 1),
                    'prescaler':(22, 1), 'rdiv2':(21, 1), 'reference_doubler':(20, 1),
                    'r_counter':(15, 5), 'clk1_divider':(3, 12)},
                3:{'reserved5':(16, 16), 'n_sel':(15, 1), 'sd_reset':(14, 1), 'reserved6':(12, 2),
                    'ramp_mode':(10, 2), 'psk_enable':(9, 1), 'fsk_enable':(8,1),
                    'ldp':(7,1), 'pd_polarity':(6, 1), 'power_down':(5, 1),
                    'cp_threestate':(4, 1), 'counter_reset':(3, 1)},
                4:{'le_sel':(31, 1), 'sd_mod_mode':(26, 5), 'reserved7':(25, 1),
                    'neg_bleed_current': (23, 2), 'readback_to_muxout': (21, 2),
                    'clk_div_mode':(19, 2), 'clk2_divider':(7, 12), 'reserved8':(3, 4)},
                5:{'reserved9':(30, 2), 'tx_ramp_clk':(29, 1), 'par_ramp':(28, 1),
                    'interrupt':(26, 2), 'fsk_ramp_en':(25, 1), 'ramp2_en':(24, 1),
                    'dev_sel':(23, 1), 'dev_offset':(19, 4), 'deviation':(3, 16)},
                6:{'reserved10':(24, 8), 'step_sel':(23, 1), 'step':(3, 20)},
                7:{'reserved11':(19, 13), 'ramp_del_fl':(18, 1), 'ramp_del':(17, 1),
                    'del_clk_sel':(16, 1), 'del_start_en':(15, 1), 'delay_start_divider':(3, 12)}
        }
        self.registers = [0]*8
        self.modified = [False]*8

        self.written_regs = [None]*8

        #Check unique names
        keys = []
        for key in self.register_def.itervalues():
            for r in key:
                if r in keys:
                    raise Exception("Duplicate register {}".format(r))
                keys.append(r)

    def freq_to_regs(self, fstart, fpd_freq, bw, length, delay):
        self.write_value(rdiv2=1)

        n = int(fstart/fpd_freq)
        frac_msb = int( ((fstart/fpd_freq) - n)*(1 << 12) )
        frac_lsb = int( (((fstart/fpd_freq) - n)*(1 << 12) - frac_msb)*(1 << 13) )

        self.write_value(n=n)
        self.write_value(frac_msb=frac_msb)
        self.write_value(frac_lsb=frac_lsb)

        clk1 = int((fpd_freq*length/(1<<20)) + 1)

        if delay > 0:
            clk1_d = int(ceil(delay*fpd_freq/(2**12)))
        else:
            clk1_d = 1

        clk1 = max(clk1, clk1_d)

        self.write_value(clk1_divider=clk1)
        self.write_value(clk2_divider=1)

        steps = int(fpd_freq*length/clk1)

        devmax = 1 << 15
        fres = fpd_freq/(1 << 25)
        fdev = bw/steps

        dev_offset = int(ceil(log(fdev/(fres*devmax), 2)))

        dev_offset = max(dev_offset, 0)

        dev = int(fdev/(fres*(1 << dev_offset)))

        self.write_value(deviation=dev)

        print 'steps', steps
        print 'clk1', clk1
        print 'fres', fres
        print 'fdev', fdev
        print 'dev_offset', dev_offset
        print 'deviation', dev

        self.write_value(step=steps)
        self.write_value(dev_offset=dev_offset)
        self.write_value(clk_div_mode=3)
        self.write_value(ramp_on=1)
        self.write_value(pd_polarity=1)
        self.write_value(prescaler=1)
        self.write_value(r_counter=1)
        self.write_value(csr_en=1)

        #Readback to muxout and negative bleed current
        #can't be activated simultaneously
        if 1:
            self.write_value(muxout=15)
            self.write_value(readback_to_muxout=3)
        else:
            self.write_value(neg_bleed_current=3)

        #Sawtooth
        self.write_value(ramp_mode=0)

        if delay > 0:
            if delay > (2**12-1)/fpd_freq:
                #Use clk1 for delay clock.
                self.write_value(del_clk_sel=1)
                d = int(round(delay*fpd_freq/clk1))
                if d > 2**12-1:
                    raise ValueError("Too large delay: {}".format(d))
                real_delay = d*clk1/fpd_freq
            else:
                #Else the delay clock is same as fpd_freq
                self.write_value(del_clk_sel=0)
                d = int(round(delay*fpd_freq))
                real_delay = d/fpd_freq

            self.write_value(ramp_del=1)
            print 'd', d
            self.write_value(delay_start_divider=d)
        else:
            real_delay = 0

        return real_delay

    def find_reg(self, reg):
        """Finds register by name"""
        for key, val in self.register_def.iteritems():
            if reg in val.keys():
                return key, val[reg]
        return None, None

    def write_value(self, **kw):
        """Write value to register, doesn't update the device"""
        for reg, val in kw.iteritems():
            #print "{} = {}".format(reg, val)
            reg_n, reg_def = self.find_reg(reg)
            if reg_n == None:
                raise ValueError("Register {} not found".format(reg))
            reg_start = reg_def[0]
            reg_len = reg_def[1]
            if val > 2**reg_len-1 or val < 0:
                raise ValueError("Invalid value, got: {}, maximum {}".format(val, 2**reg_len-1))
            #Clear previous value
            self.registers[reg_n] &= (~((((2**reg_len-1))&0xFFFFFFFF) << reg_start) & 0xFFFFFFFF)
            self.registers[reg_n] |= (val) << reg_start
            self.modified[reg_n] = True
        return

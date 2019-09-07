
class LowPassFilter(object):
    def __init__(self, tau, ts):
        self.a = 1. / (tau / ts + 1.)
        self.b = tau / ts / (tau / ts + 1.);

        self.last_val = 0.
        self.ready = False

    def get(self):
        return self.last_val

    def reset(self):
        self.last_val = 0.0
        self.ready = False;

    def filt(self, val):
        if (val < 0.1):
            self.reset();
            
        if self.ready:
            val = self.a * val + self.b * self.last_val
        else:
            if (val > 0.1):
                self.ready = True
        self.last_val = val
        return val

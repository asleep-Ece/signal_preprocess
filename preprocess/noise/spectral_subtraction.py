import scipy
import numpy as np


class SpectralSubtraction:
    def __init__(self, winsize, window, coefficient=5.0, ratio=1.0):
        self._window = window
        self._coefficient = coefficient
        self._ratio = ratio

    def compute(self, signal, noise):
        n_spec = scipy.fft(noise * self._window)
        n_pow = scipy.absolute(n_spec) ** 2.0
        return self.compute_by_noise_pow(signal, n_pow)

    def compute_by_noise_pow(self, signal, n_pow):
        s_spec = scipy.fft(signal * self._window)
        s_amp = scipy.absolute(s_spec)
        s_phase = scipy.angle(s_spec)
        s_amp2 = s_amp**2.0
        amp = s_amp2 - n_pow * self._coefficient

        amp = scipy.maximum(amp, 0.01 * s_amp2)
        amp = scipy.sqrt(amp)
        amp = self._ratio * amp + (1.0 - self._ratio) * s_amp
        spec = amp * scipy.exp(s_phase * 1j)
        return scipy.real(scipy.ifft(spec))


def get_frame(signal, winsize, no):
    shift = winsize / 2
    start = int(no * shift)
    end = start + winsize
    return signal[start:end]


def add_signal(signal, frame, winsize, no):
    shift = winsize / 2
    start = int(no * shift)
    end = start + winsize
    signal[start:end] = signal[start:end] + frame


def spectral_subtraction(signal, noise_clip, winsize=2**10, window=scipy.hanning(2**10)):
    """Reduce noise."""
    method = SpectralSubtraction(winsize, window)

    out = scipy.zeros(len(signal), scipy.float32)
    power = (
        scipy.signal.welch(noise_clip, window=window, return_onesided=False, scaling="spectrum")[1] * window.sum() ** 2
    )
    nf = int(len(signal) / (winsize / 2) - 1)
    for no in range(nf):
        s = get_frame(signal, winsize, no)
        add_signal(out, method.compute_by_noise_pow(s, power), winsize, no)
    return out

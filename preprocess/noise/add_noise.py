import numpy as np


def add_gaussian_noise(signal, noise_factor):
    noise = np.random.randn(len(signal))  # radn generate a sample from "standard normal" distribution
    augmented_signal = signal + noise_factor * noise

    # Cast back to same signal type
    augmented_signal = augmented_signal.astype(type(signal[0]))
    return augmented_signal

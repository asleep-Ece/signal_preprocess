import numpy as np
from soundsleep.preprocess.utils import framing_signal, short_time_energy


def noise_minimum_energy(signal, sr, framesize=0.1, overlap=0.05):
    # Framing signals
    frames, _ = framing_signal(signal, sr, framesize=framesize, overlap=overlap)

    # Get frame energy vals
    frame_energy_vals = short_time_energy(frames)

    # Noise clip is the frame with minimum energy
    minimum_energy_idx = np.argmin(frame_energy_vals)
    noise_clip = frames[minimum_energy_idx]

    return noise_clip

import os
from datetime import datetime

import librosa
import numpy as np

from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating
from soundsleep.preprocess.utils import visualize, write_wav

if __name__ == "__main__":
    IN_FILE = None  # TODO: write file name here
    OUT_PATH = None  # TODO: set output path
    os.makedirs(OUT_PATH, exist_ok=True)

    # Loading audio assuming that the audio is an uncompressed WAV file
    print("Loading audio...")
    signal, sr = librosa.load(IN_FILE, sr=None)

    print("Start noise reduction!")
    start = datetime.now()
    nr_sig = adaptive_noise_reduce(
        signal, sr, segment_len=10, estimate_noise_method=noise_minimum_energy, reduce_noise_method=spectral_gating
    )
    print("Time elapsed: {}".format(datetime.now() - start))

    # Write audio file
    audio_path = os.path.join(OUT_PATH, "nr_audio.wav")
    write_wav(nr_sig, sr, audio_path)

    # Visualize
    fig_path = os.path.join(OUT_PATH, "fig.png")
    s_idx = 4 * 60 * 60 * sr
    e_idx = s_idx + 60 * sr
    signal_list = [signal[s_idx:e_idx], nr_sig[s_idx:e_idx]]
    title_list = ["Original audio", "Noise-reduced audio"]
    visualize(signal_list, title_list, save=True, path=fig_path)

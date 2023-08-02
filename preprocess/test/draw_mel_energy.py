from genericpath import exists
import os

import numpy as np
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt

from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating


def get_energy(mel):
    mel = np.transpose(mel)  # (20, 1021) -> (1201, 20)
    energy = np.sum(mel**2, axis=1)
    energy = librosa.power_to_db(energy, ref=np.max)
    return energy


def draw_mel_energy(mel_data, stage_name, path=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [7, 1]}, sharex=True)

    # Mel Spectrogram
    mel_db = librosa.power_to_db(mel_data, ref=np.max)
    specshow(mel_db, sr=16000, x_axis="time", y_axis="mel", ax=axes[0], hop_length=400)
    axes[0].set_xlabel(None)

    # Energy
    energy = get_energy(mel_data)
    x = np.linspace(0, 30, len(energy))
    axes[1].plot(x, energy, "k-")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("dB")

    fig.tight_layout()
    plt.savefig(path + ".png", dpi=300)
    plt.savefig(path + ".pdf")
    plt.savefig(path + ".svg")
    plt.close()


if __name__ == "__main__":
    key = "4c"
    ID = "1002"
    stage_name_dict = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM"}
    save_path = "{}_mels_energy".format(ID)
    os.makedirs(save_path, exist_ok=True)
    data_path = "/mnt/SSD/.data/sam2sam_19_20_final_2/data1/test"
    file_list = [os.path.join(data_path, x) for x in os.listdir(data_path) if "{}_data".format(ID) in x]

    for file in file_list:
        data = np.load(file, allow_pickle=True).item()
        stage_name = stage_name_dict[data[key]]
        mel_data = data["x"]
        energy = get_energy(mel_data)
        index = int(os.path.splitext(file)[0].split("_")[-1])
        save_file = os.path.join(save_path, "{}_{}".format(index, stage_name))
        draw_mel_energy(mel_data, stage_name, save_file)

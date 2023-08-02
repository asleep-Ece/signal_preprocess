import os

import librosa
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
import random
from soundsleep.utils.spec_analysis import estimate_SNR

stage_dict = {0: "WAKE", 1: "NREM", 2: "REM"}


def draw(spec_db, spec_SNR_all, energy_all, filename, n, visualize=False):
    fig, axes = plt.subplots(n, n, sharex=True, sharey=True, figsize=(15, 10))
    for i in range(n):
        for j in range(n):
            specshow(spec_db[i * n + j], sr=16000, x_axis="time", y_axis="mel", ax=axes[i, j], hop_length=400)

    fig2, axes2 = plt.subplots(n, n, sharex=True, sharey=True, figsize=(15, 10))
    for i in range(n):
        for j in range(n):
            axes2[i, j].plot(energy_all[i * n + j])

    fig.suptitle("Mel Spectrogram - {} {}".format(os.path.basename(filename), spec_SNR_all))
    fig.savefig(filename + "_{}.png".format(int(n**2)))

    if visualize:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    # Set paths
    IN_PATH = "/mnt/SSD/.data/sam2sam_19_20/data1/test"  # TODO: set path to npy files
    OUT_PATH = "DRAW_MELS"  # TODO: set output path
    os.makedirs(OUT_PATH, exist_ok=True)

    # Configurations
    N_CLASS = 3
    key = "{}c".format(N_CLASS)
    N = 9  # number of mel spectrograms to draw

    # Draw
    for i in range(N_CLASS):
        cnt = 0
        spec_db_all = []
        spec_SNR_all = []
        energy_all = []

        list_dir = os.listdir(IN_PATH)
        random.shuffle(list_dir)
        for file in list_dir:
            file = os.path.join(IN_PATH, file)
            data = np.load(file, allow_pickle=True).item()
            if data[key] == i:
                spec_db = librosa.power_to_db(data["x"], ref=np.max)
                # spec_db = librosa.power_to_db(data['x'])
                spec_SNR, energy = estimate_SNR(data["x"])

                spec_db_all.append(spec_db)
                spec_SNR_all.append(round(spec_SNR, 2))
                energy_all.append(energy)
                cnt += 1
                if cnt == N:
                    break
        filename = os.path.join(OUT_PATH, stage_dict[i])
        draw(np.array(spec_db_all), spec_SNR_all, energy_all, filename, int(np.sqrt(N)), True)

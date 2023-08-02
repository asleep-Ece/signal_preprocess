import os

import torch
import librosa
from librosa.display import specshow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from soundsleep.datasets.transforms import (
    RandomPitch,
    SeqRandomPitch,
    RandomLoudness,
    SeqRandomLoudness,
    RandomDynamic,
    SeqRandomDynamic,
    RandomNoise,
    RandomSpecAugment,
    SeqRandomSpecAugment,
)

# matplotlib.use('Agg')


def draw(old_mel, new_mel, save=False, path=None, show=False, index=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Old mel
    old_spec_db = librosa.power_to_db(old_mel, ref=np.max)
    specshow(old_spec_db, sr=16000, x_axis="time", y_axis="mel", ax=axes[0], hop_length=400, fmax=8000)
    axes[0].set_title("Old Mel Spectrogram")

    # New mel
    new_spec_db = librosa.power_to_db(new_mel, ref=np.max)
    specshow(new_spec_db, sr=16000, x_axis="time", y_axis="mel", ax=axes[1], hop_length=400, fmax=8000)
    axes[1].set_title("New Mel Spectrogram")

    fig.tight_layout()

    if save:
        if index:
            path = str(index) + "_fig.png" if path == None else os.path.join(path, str(index) + "_fig.png")
        else:
            path = "fig.png" if path == None else os.path.join(path, "fig.png")
        plt.savefig(path)
    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    IS_SAMPLE = True

    if IS_SAMPLE:
        FILE_PATH = "/mnt/SSD/.data/sam2sam_19_20_final_2/data1/test"
        FILE_NAME = "1031_data_0_100.npy"
        OUT_PATH = "VISUALIZE_TRANSFORMED_MEL"

        os.makedirs(OUT_PATH, exist_ok=True)

        filename = os.path.join(FILE_PATH, FILE_NAME)
        data = np.load(filename, allow_pickle=True).item()

        old_mel = data["x"]
        # transform = RandomPitch(0.2, verbose=True)
        # transform = RandomLoudness([1e4], verbose=True)
        # transform = RandomDynamic([2], verbose=True)
        transform = RandomNoise([1e-6], verbose=True)
        # transform = RandomSpecAugment()
        new_mel = transform(old_mel)

        if 0:
            shape = old_mel.shape
            old_mel = np.reshape(old_mel, (-1, shape[0], shape[1]))
            old_mel = torch.from_numpy(old_mel)
            new_mel = spec_augment_pytorch.spec_augment(mel_spectrogram=old_mel)
            # new_mel = np.ascontiguousarray(np.swapaxes(new_mel, 0, 1))
            old_mel = old_mel[0, :, :]
            new_mel = new_mel[0, :, :]

            print("new_mel shape: ", new_mel.shape)

        if 0:
            # Convert to dB scale
            new_mel = [librosa.power_to_db(xi) for xi in new_mel]
            # new_mel = [librosa.power_to_db(xi, ref=np.max) for xi in new_mel]
            new_mel = np.array(new_mel)

        if 0:
            # Normalize each Mel Spectrogram in the sequence
            for i in range(new_mel.shape[0]):
                mean = new_mel[i].mean()
                std = new_mel[i].std()
                new_mel[i] = (new_mel[i] - mean) / std

        draw(old_mel, new_mel, save=True, path=OUT_PATH, show=True)

    else:
        FILE_PATH = "/mnt/SSD/.data/sam2sam_19_20_final_2/data1/test"
        FILE_NAME = "1031_data_0_100.npy"
        OUT_PATH = "VISUALIZE_TRANSFORMED_MEL"

        os.makedirs(OUT_PATH, exist_ok=True)

        filename = os.path.join(FILE_PATH, FILE_NAME)
        data = np.load(filename, allow_pickle=True).item()

        old_mel = data["x"]
        old_mel = np.tile(old_mel, (10, 1, 1))
        # transform = SeqRandomLoudness(verbose=True)
        # transform = SeqRandomPitch(0.2, verbose=True)
        # transform = SeqRandomDynamic(verbose=True)
        # transform = RandomNoise(verbose=True)
        transform = SeqRandomSpecAugment(verbose=True)
        new_mel = transform(old_mel)

        for i in range(old_mel.shape[0]):
            draw(old_mel[i], new_mel[i], save=True, path=OUT_PATH, show=True, index=i)

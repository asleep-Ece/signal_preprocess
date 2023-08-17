import os

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

from soundsleep.datasets.utils import data_balancing
from soundsleep.datasets.transforms import RandomPitch


class SNUBH(Dataset):
    """Dataset Class which holds sound sleep data of patients."""

    def __init__(self, mode, input_paths, n_classes, augment=False, oversample_data=True, *args, **kwargs):
        # Get list of all files
        self.file_list = []
        for input_path in input_paths:
            for dir_ in os.listdir(input_path):
                data_path = os.path.join(input_path, dir_, mode)
                if not os.path.exists(data_path):
                    continue
                all_file = [os.path.join(data_path, x) for x in os.listdir(data_path) if "stats" not in x]
                self.file_list += all_file

        # Label key to get the corresponding label from data dictionary
        self.label_key = "{}c".format(n_classes)

        # Data augmentation
        self.transform = RandomPitch(0.2) if (augment and mode == "train") else None

        if mode == "train" and oversample_data:
            # Data balancing by data oversampling
            self.file_list = data_balancing(self.file_list, n_classes, sequence=False, verbose=True)

        self.len = len(self.file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get file
        file_path = self.file_list[idx]

        # Load data
        data_dict = np.load(file_path, allow_pickle=True).item()
        x = data_dict["x"]
        y = data_dict[self.label_key]

        # Apply data augmentation if defined
        if self.transform:
            if np.random.randint(2) == 1:
                x = self.transform(x)

        # Convert to dB scale
        x = librosa.power_to_db(x, ref=np.max)

        # Normalize the Mel Spectrogram data
        mean = x.mean()
        std = x.std()
        x = (x - mean) / std

        # Input reverse (refer to "Sequence to Sequence Learning with Neural Network" paper)
        x = x[:, ::-1]

        # Transpose (n_mels, time) -> (time, n_mels)
        x = np.ascontiguousarray(np.swapaxes(x, 0, 1))

        # Convert to Torch tensors
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        return x, y

import logging
import os
import warnings
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from soundsleep.preprocess.data_generator import BaseDataGenerator
from soundsleep.preprocess import utils
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating
from soundsleep.preprocess.end2end_preprocess import load_audio_list


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--sample_time", type=int, default=30, metavar="N", help="Time in one sample (in seconds)")
    parser.add_argument(
        "--segment_len",
        type=float,
        default=30,
        metavar="N",
        help="Length of each segment in which the noise is estimated",
    )
    parser.add_argument(
        "--nbytes", type=int, default=2, metavar="N", help="Number of bytes to encode one sample in the audio"
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        metavar="N",
        help="Smoothing coefficient to smooth the transition between different noise estimations",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=20,
        metavar="N",
        help="Number of evenly spaced frequencies to separate into in Mel Spectrogram",
    )
    parser.add_argument(
        "--window_size",
        type=float,
        default=50e-3,
        metavar="N",
        help="Window size which determines n_fft in librosa.feature.melspectrogram",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=25e-3,
        metavar="N",
        help="Stride which determines hop_length in librosa.feature.melspectrogram",
    )
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio data")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the preprocessed data")
    parser.add_argument("-nroff", "--noise_reduce_off", action="store_true", help="Turn off noise reduction")
    parser.add_argument(
        "--upper_bound_zero_portion",
        type=float,
        default=0.15,
        metavar="N",
        help="Maximum allowed portion of zero values in audio signal (default: 15%)",
    )
    parser.add_argument(
        "--lower_bound_n_audio_files",
        type=int,
        default=48,
        metavar="N",
        help="Minimum number of audio files to get preprocessed (default: 48 - 4 hours)",
    )
    parser.add_argument(
        "--upper_bound_n_audio_files",
        type=int,
        default=180,
        metavar="N",
        help="Maximum number of audio files to get preprocessed (default: 180 - 15 hours)",
    )
    return parser


def get_audio_idx_list(file_list) -> List:
    """Get audio index list."""
    audio_idx_list = []
    for file in file_list:
        file = os.path.basename(file)
        audio_idx = int(file.split("_")[0])
        if audio_idx not in audio_idx_list:
            audio_idx_list.append(audio_idx)
        else:
            print("Duplicated index: {}".format(file))
    return sorted(audio_idx_list)


class SmartphoneUnsupervised(BaseDataGenerator):
    def __init__(self, parser: ArgumentParser) -> None:
        super(SmartphoneUnsupervised, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

        # Audio path
        self.audio_path = self.args.audio_path
        print(f"audio_path: {self.audio_path}")

        # Output path to save npy files
        if self.args.audio_to_numpy:
            self.output_path = os.path.join(self.args.output_path, "data")
            self.matching_out_csv = os.path.join(self.output_path, "matching_data.csv")
            os.makedirs(self.output_path, exist_ok=False)
            print(f"output_path: {self.output_path}")

    def _update_idx(self, data_idx: int, mel_file_idx: int, data_stride: int) -> Tuple[int, int]:
        data_idx += data_stride
        mel_file_idx += 1
        return data_idx, mel_file_idx

    def audio2numpy(self) -> None:
        """Preprocessing audio data and saved in numpy format."""

        logging.basicConfig(
            filename=os.path.join(self.output_path, "preprocess.log"),
            level=logging.CRITICAL,
        )

        print("Getting the list of directories for sleep sessions")
        train_list = utils.retrieve_mobile_data_paths(self.audio_path)
        print(f"Number of sleep sessions for training: {len(train_list)}")

        print("\nData extraction for training set...")
        train_path = os.path.join(self.output_path, "train")
        self.extract_data(train_list, train_path, self.args)

        print(f"\nData saved in {self.output_path}.")

    def extract_data(self, session_dir_list: List[str], output_path: str, args: Namespace) -> None:
        """Get audio and label data, then save them into output_path in npy format.

        Args:
            session_dir_list: List of paths to the sleep session data (folder).
            output_path: Path to save npy files.
            args: Arguments for preprocessing.
        """

        # Matching data to be used in the future
        df = pd.DataFrame(columns=["ID", "Email", "Recorded datetime", "Zero portion"])

        os.makedirs(output_path, exist_ok=False)

        for i, session_dir in enumerate(tqdm(session_dir_list, desc="Preprocessing data", ncols=120)):
            # Get all available audios
            audio_list = []
            for x in os.listdir(session_dir):
                if x.endswith(".mp3") or x.endswith(".aac"):
                    audio_list.append(os.path.join(session_dir, x))

            # Check if number of audio files is enough
            if len(audio_list) < self.args.lower_bound_n_audio_files:
                log = f"[{session_dir}] Number of audio files is not enough: len(audio_list) = {len(audio_list)}."
                print(log)
                logging.critical(log)
                continue

            # Check if number of audio files exceeds the limit
            if len(audio_list) > self.args.upper_bound_n_audio_files:
                log = f"[{session_dir}] Number of audio files is too many: len(audio_list) = {len(audio_list)}."
                print(log)
                logging.critical(log)
                continue

            # Sort the audio_list
            audio_list = sorted(audio_list, key=lambda x: int(x.split("/")[-1].split("_")[0]))

            # Get list of audio file indices
            audio_idx_list = get_audio_idx_list(audio_list)
            if len(audio_list) != len(audio_idx_list):
                log = f"[{session_dir}] len(audio_list) = {len(audio_list)} while len(audio_idx_list) = {len(audio_idx_list)}"
                print(log)
                logging.critical(log)
                continue

            # Safety check if the files are in perfect sequence
            if (np.max(audio_idx_list) - np.min(audio_idx_list) + 1) != len(audio_idx_list):
                log = "[{}] Audio data is not contiguous. min_id = {}, max_id = {}, n_file = {}".format(
                    session_dir, np.min(audio_idx_list), np.max(audio_idx_list), len(audio_idx_list)
                )
                print(log)
                logging.critical(log)
                continue

            # Read audio data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    all_audio_data, sr = load_audio_list(audio_list, ffmpeg=False)
            except EOFError as e:
                log = "[{}] {}".format(session_dir, e.__context__)
                print(log)
                logging.critical(log)
                continue

            # 8k audio is normally usable but better be removed for Smartphone audio data
            if sr == 8000:
                log = "[{}] sampling_rate = {}".format(session_dir, sr)
                print(log)
                logging.critical(log)
                continue

            # Get zero portion in audio signal
            zero_portion = utils.get_zero_portion(all_audio_data)
            if zero_portion > args.upper_bound_zero_portion:
                log = "[{}] Audio data has {:.2%} zero values, which is more than the allowed threshold: {:.2%}".format(
                    session_dir, zero_portion, args.upper_bound_zero_portion
                )
                print(log)
                logging.critical(log)
                continue

            # Noise reduction if applicable
            if args.noise_reduce_off:
                pass
            else:
                all_audio_data = adaptive_noise_reduce(
                    all_audio_data,
                    sr,
                    args.segment_len,
                    estimate_noise_method=noise_minimum_energy,
                    reduce_noise_method=spectral_gating,
                    smoothing=args.smoothing,
                )

            # Prepare important constants for data extraction
            sample_len = int(args.sample_time * sr)
            data_stride = int(30 * sr)

            # Read and extract data
            data_idx = 0  # Data reading starts from 0 index
            mel_file_idx = 0  # Mel file index starts from 0
            audio_file_idx = 0  # Audio file index starts from 0
            data_len = len(all_audio_data)  # Data length
            not_zero_sum_mel = True  # Current summation of Mel element being 0 or not

            while True:
                # Break if the end of the audio is reached
                if (data_idx + sample_len) > data_len:
                    break

                # A segment of 30 second sound
                audio_30s = all_audio_data[data_idx : data_idx + sample_len]

                # Convert into mel spectrogram
                mel_spec = utils.mel_spectrogram(audio_30s, sr, args.n_mels, args.window_size, args.stride)

                # Check if Mel spectrogram is empty, then change indices if necessary
                if mel_spec.sum() == 0:
                    data_idx += data_stride  # Next 30-second data
                    if not_zero_sum_mel:
                        audio_file_idx += 1  # Consider the next epoch as a next audio file
                        mel_file_idx = 0  # Reset Mel file index to 0 for the next audio file
                        not_zero_sum_mel = False  # Turn to False because sum of Mel data is 0 now
                    continue

                # Turn not_zero_sum_mel flag to True again if it was False
                if not not_zero_sum_mel:
                    not_zero_sum_mel = True  # Not zero sum anymore

                # Save mel-spectrogram into the npy file
                data_dict = {"x": mel_spec, "is_8k": False}
                file_name = os.path.join(output_path, "{:03d}_data_{}_{}.npy".format(i, audio_file_idx, mel_file_idx))
                np.save(file_name, data_dict)

                data_idx, mel_file_idx = self._update_idx(data_idx, mel_file_idx, data_stride)

            del all_audio_data

            # Adding data to the matching CSV file
            session_info = session_dir.split("/")[-2:]  # [email, datetime_xxxx]
            email = session_info[0]
            recorded_datetime = session_info[1].split("_")[0]
            row_data = ["{:03d}".format(i), email, recorded_datetime, "{:.2%}".format(zero_portion)]
            df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

            # Save the matching data into a CSV file
            df.to_csv(self.matching_out_csv, index=False)

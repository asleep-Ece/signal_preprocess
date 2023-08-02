from argparse import ArgumentParser, Namespace
import logging
import os
from typing import List, Tuple
import warnings

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


class HomeSleepNetMbrainData(BaseDataGenerator):
    def __init__(self, parser: ArgumentParser) -> None:
        super(HomeSleepNetMbrainData, self).__init__()

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

    def _update_idx(self, data_idx: int, file_idx: int, data_stride: int) -> Tuple[int, int]:
        data_idx += data_stride
        file_idx += 1
        return data_idx, file_idx

    def audio2numpy(self) -> None:
        """Preprocessing audio data and saved in numpy format."""

        logging.basicConfig(
            filename=os.path.join(self.output_path, "preprocess.log"),
            level=logging.CRITICAL,
        )

        train_list = utils.retrieve_mobile_data_paths(self.audio_path)

        print("\nData extraction for training set...")
        train_path = os.path.join(self.output_path, "train")
        self.extract_data(train_list, train_path, self.args)

        print(f"\nData saved in {self.output_path}.")

    def get_longest_continuous(self, email_day_dir_list: List) -> str:
        """Returns the longest session with continuous audio files.

        Finds and returns the longest session with continuous audio files
        from the sessions in `email_day_dir_list`."""

        n_file_list = []
        continuous_list = []
        n_files_6hours = int(6 * 60 / 5)
        for email_day_dir in email_day_dir_list:
            audio_list = [x for x in os.listdir(email_day_dir) if x.endswith(".mp3")]
            n_file_list.append(len(audio_list))

            if len(audio_list) == 0:
                continuous_list.append(False)
                continue

            # Safety check if the files are in perfect sequence
            audio_idx_list = get_audio_idx_list(audio_list)
            if (np.max(audio_idx_list) - np.min(audio_idx_list) + 1) != len(audio_list):
                continuous_list.append(False)
            else:
                continuous_list.append(True)

        n_file_list = np.array(n_file_list)
        continuous_list = np.array(continuous_list)

        # Length of sessions with continuous audio files
        n_file_continuous_list = n_file_list[continuous_list == True]
        if len(n_file_continuous_list) == 0:
            return "NoContinuous"

        # Length of the longest session with continuous audio files
        n_longest_continuous = max(n_file_continuous_list)
        if n_longest_continuous < n_files_6hours:
            return "LessThan6Hours"

        # Find the longest session name with continuous audio files
        n_longest_continuous_index = np.where(n_file_list == n_longest_continuous)[0][0]
        longest_continuous_day_dir = email_day_dir_list[n_longest_continuous_index]

        return longest_continuous_day_dir

    def extract_data(self, day_dir_list: List[str], output_path: str, args: Namespace) -> None:
        """Get audio and label data, then save them into output_path in npy format.

        Args:
            day_dir_list: List of paths to the day data (folder).
            output_path: Path to save npy files.
            args: Arguments for preprocessing.
        """

        # Matching data to be used in the future
        df = pd.DataFrame(columns=["ID", "Email", "Recorded datetime", "Zero portion"])

        # Getting all email list
        email_list = [x for x in os.listdir(args.audio_path) if os.path.isdir(os.path.join(args.audio_path, x))]

        os.makedirs(output_path, exist_ok=False)

        for i, email in enumerate(tqdm(email_list, desc="Preprocessing data", ncols=120)):
            # Filters and gets day directories from the `email`
            email_day_dir_list = [x for x in day_dir_list if email in x]

            # Get only one longest session with continuous audio files
            day_dir = self.get_longest_continuous(email_day_dir_list)

            if day_dir == "NoContinuous":
                log = f"[{email}] This user has no sessions with continuous audio files."
                print(log)
                logging.critical(log)
                continue
            elif day_dir == "LessThan6Hours":
                log = f"[{email}] This user has no sessions with continuous audio files longer than 6 hours."
                print(log)
                logging.critical(log)
                continue
            else:  # Normal case
                pass

            # Get audio list
            audio_list = [os.path.join(day_dir, x) for x in os.listdir(day_dir) if x.endswith(".mp3")]

            # Check if the audio list is empty
            if len(audio_list) == 0:
                log = f"[{day_dir}] len(audio_list) = 0"
                print(log)
                logging.critical(log)
                continue

            # Sort the audio_list
            audio_list = sorted(audio_list, key=lambda x: int(x.split("/")[-1].split("_")[0]))

            # Read audio data
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    all_audio_data, sr = load_audio_list(audio_list, ffmpeg=False)
            except EOFError as e:
                log = "[{}] {}".format(day_dir, e.__context__)
                print(log)
                logging.critical(log)
                continue

            # Get zero portion in audio signal
            zero_portion = utils.get_zero_portion(all_audio_data)
            if zero_portion > args.upper_bound_zero_portion:
                log = "[{}] Audio data has {:.2%} zero values, which is more than the allowed threshold: {:.2%}".format(
                    day_dir, zero_portion, args.upper_bound_zero_portion
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
            data_idx = 0
            file_idx = 0
            data_len = len(all_audio_data)

            while True:
                # Break if the end of the audio is reached
                if (data_idx + sample_len) > data_len:
                    break

                # A segment of 30 second sound
                audio_30s = all_audio_data[data_idx : data_idx + sample_len]

                # Convert into mel spectrogram
                mel_spec = utils.mel_spectrogram(audio_30s, sr, args.n_mels, args.window_size, args.stride)

                # Save mel-spectrogram into the npy file
                data_dict = {"x": mel_spec}
                file_name = os.path.join(output_path, "{:03d}_data_0_{}.npy".format(i, file_idx))
                np.save(file_name, data_dict)

                data_idx, file_idx = self._update_idx(data_idx, file_idx, data_stride)

            del all_audio_data

            # Adding data to the matching CSV file
            day_name = day_dir.split("/")[-2:]  # [email, datetime_xxxx]
            email = day_name[0]
            recorded_datetime = day_name[1].split("_")[0]
            row_data = ["{:03d}".format(i), email, recorded_datetime, "{:.2%}".format(zero_portion)]
            df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

            # Save the matching data into a CSV file
            df.to_csv(self.matching_out_csv, index=False)

import os
import logging
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import librosa

from soundsleep.preprocess.data_generator import BaseDataGenerator
from soundsleep.preprocess import utils
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating
from soundsleep.preprocess.end2end_preprocess import round_data


def add_arguments(parser):
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
    parser.add_argument("--psg_path", type=str, required=True, help="Path to the corresponding PSG data")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the preprocessed data")
    parser.add_argument("-nroff", "--noise_reduce_off", action="store_true", help="Turn off noise reduction")
    return parser


class SNUBHMinihexa(BaseDataGenerator):
    def __init__(self, parser):
        super(SNUBHMinihexa, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

        # Audio path
        self.audio_path = self.args.audio_path
        print("audio_path: {}".format(self.audio_path))

        # Output path to save npy files
        if self.args.audio_to_numpy:
            self.output_path = os.path.join(self.args.output_path, "data", "test")
            self.matching_out_csv = os.path.join(self.args.output_path, "matching_data.csv")
            os.makedirs(self.output_path, exist_ok=False)
            print("output_path: {}".format(self.output_path))

    def audio2numpy(self):
        """Preprocessing audio data and saved in numpy format."""

        logging.basicConfig(filename=os.path.join(self.args.output_path, "preprocess.log"), level=logging.CRITICAL)

        # Get a list to all available Minihexa recordings
        recording_list = get_list_recordings(self.audio_path)

        # Data extraction
        print("Data extraction process started!")
        extract_data(recording_list, self.output_path, self.matching_out_csv, self.args)


def get_list_recordings(audio_path):
    """Return a list of all absolute paths to all the Minihexa recordings."""
    recording_list = []
    for room in os.listdir(audio_path):
        if "room" not in room:
            continue
        full_room = os.path.join(audio_path, room)
        for date_time in os.listdir(full_room):
            full_date_time = os.path.join(full_room, date_time)
            if not os.path.isdir(full_date_time) or ".DS_" in date_time or "MACRO" in date_time:
                continue
            recording_list.append(full_date_time)
    print("Total number of recordings: {}".format(len(recording_list)))
    return recording_list


def get_audio_start_time(record_dir):
    start_time = os.path.basename(record_dir)[8:]
    start_time = start_time[:4] + ":" + start_time[4:]
    start_time = start_time[:2] + ":" + start_time[2:]
    return start_time


def get_corresponding_psg_dir(record_dir, psg_path):
    room = record_dir.split("/")[-2]
    date = record_dir.split("/")[-1][2:8]
    psg_dir = os.path.join(psg_path, "{}_{}".format(room, date))
    return psg_dir


def load_audio_list(record_dir):
    """Return data loaded from the list of Minihexa recorded audios."""
    # List and sort all files
    audio_files = [x for x in os.listdir(record_dir) if ".wav" in x]
    audio_files = sorted(audio_files, key=lambda x: int(x.split("_")[0]))
    audio_files = [os.path.join(record_dir, x) for x in audio_files]

    if len(audio_files) == 0:
        log = "[{}] There are no audio files.".format(record_dir)
        print(log)
        logging.critical(log)
        return [], 0

    # Load audio
    audio_data, sr = [], 0
    out_file = "16kHz_audio.wav"
    for audio_file in audio_files:
        try:
            # Convert to 16 kHz audio and load
            subprocess.run(
                ["ffmpeg", "-i", audio_file, "-ar", "16000", out_file],
                timeout=90,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            log = "[{}] ffmpeg conversion error happened when converting to 16 kHz: {}".format(record_dir, e)
            print(log)
            logging.critical(log)
            if os.path.exists(out_file):
                os.remove(out_file)
            return [], 0

        # Load converted 16kHz audio
        audio, sr = librosa.load(out_file, sr=None)
        assert sr == 16000
        audio = round_data(audio, length=5 * 60 * sr)
        os.remove(out_file)
        audio_data.append(audio)
    audio_data = np.array(audio_data).reshape(-1)

    return audio_data, sr


def extract_data(recording_list, output_path, matching_out_csv, args):
    """Get audio and label data, then save them into output_path in npy format."""

    # Matching data to be used in the future
    df = pd.DataFrame(columns=["Index", "Minihexa audio path", "PSG path"])

    n_recordings = len(recording_list)
    start_time = datetime.now()

    for i, record_dir in enumerate(recording_list):
        start = datetime.now()
        psg_dir = get_corresponding_psg_dir(record_dir, args.psg_path)
        if not os.path.exists(psg_dir):
            log = "[{}] Corresponding PSG path {} does not exist.".format(record_dir, psg_dir)
            print(log)
            logging.critical(log)
            continue

        try:
            data_offset = utils.calculate_data_offset(psg_dir)
            extract_label = utils.extract_label_file(psg_dir)
        except (RuntimeError, ValueError) as e:
            log = "[{}] {}".format(os.path.basename(psg_dir), e)
            print(log)
            logging.critical(log)
            continue

        if data_offset == False or extract_label == False:
            log = "[{}] psg_dir = {}, data_offset = {}, extract_label = {}. Skipping.".format(
                record_dir, psg_dir, data_offset, extract_label
            )
            print(log)
            logging.critical(log)
            continue

        label_file = os.path.join(psg_dir, "sleep_labels.csv")
        offset_file = os.path.join(psg_dir, "data_offset.csv")
        try:
            LABEL_START = pd.read_csv(offset_file)["label_start"].values[0]
        except (RuntimeError, IndexError) as e:
            log = "[{}] {}: cannot extract LABEL_START from the offset file: {}".format(record_dir, psg_dir, e)
            print(log)
            logging.critical(log)
            continue

        # Load label
        labels_data = utils.read_labels(label_file)
        if len(labels_data) == 0:
            log = "[{}] {}: No labels data.".format(record_dir, psg_dir)
            print(log)
            logging.critical(log)
            continue

        # Load audio
        audio_data, sr = load_audio_list(record_dir)
        if len(audio_data) == 0:
            continue
        assert sr == 16000

        # Synchronize the audio with the label
        AUDIO_START = get_audio_start_time(record_dir)
        indices = utils.sync_audio_label(len(audio_data), sr, len(labels_data), AUDIO_START, LABEL_START, record_dir)
        if (type(indices) == bool) and (indices == False):
            continue
        else:
            audio_data = audio_data[indices[0] : indices[1]]
            labels = labels_data[indices[2] : indices[3]]

        # Noise reduction
        if args.noise_reduce_off:
            pass
        else:
            audio_data = adaptive_noise_reduce(
                audio_data,
                sr,
                args.segment_len,
                estimate_noise_method=noise_minimum_energy,
                reduce_noise_method=spectral_gating,
                smoothing=args.smoothing,
            )

        # Prepare important constants for data extraction
        SAMPLE_LEN = int(args.sample_time * sr)
        DATA_STRIDE = int(30 * sr)
        LABEL_STRIDE = 1
        N_HEAD_TAIL = int((args.sample_time - 30) / 2 / 30)

        def update_idx(data_idx, label_idx, file_idx):
            data_idx += DATA_STRIDE
            label_idx += LABEL_STRIDE
            file_idx += 1
            return data_idx, label_idx, file_idx

        # Read and extract data
        data_idx = 0
        label_idx = N_HEAD_TAIL
        file_idx = 0
        data_len = len(audio_data)

        while True:
            # Break condition
            if (data_idx + SAMPLE_LEN) > data_len:
                assert (data_idx / (30 * sr) + N_HEAD_TAIL) == label_idx
                break

            # Get labels
            y_5c = labels[label_idx]
            y_2c, y_3c, y_4c = utils.convert_stage(y_5c)

            # Extract data
            audio_30sec = audio_data[data_idx : data_idx + SAMPLE_LEN]

            # Mel Spectrogram
            mel_spec = utils.mel_spectrogram(audio_30sec, sr, args.n_mels, args.window_size, args.stride)

            # Save into a npy file in the dictionary format
            data_dict = {"x": mel_spec, "2c": y_2c, "3c": y_3c, "4c": y_4c, "5c": y_5c}
            file_name = os.path.join(output_path, "{:03d}_data_0_{}.npy".format(i, file_idx))
            np.save(file_name, data_dict)

            # Update indices
            data_idx, label_idx, file_idx = update_idx(data_idx, label_idx, file_idx)

        del audio_data

        if ((i + 1) % 10 == 0) or (i == 0) or (i == n_recordings - 1):
            now = datetime.now()
            time = now - start
            elapsed = now - start_time
            expected = time * n_recordings
            print(
                "[{}/{}] num_samples = {}, time = {}, time_elapsed = {}, time_expected = {}".format(
                    i + 1, n_recordings, file_idx + 1, time, elapsed, expected
                )
            )

        # Adding data to the matching data
        row_data = ["{:03d}".format(i), record_dir, psg_dir]
        df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

    # Save the matching data into a CSV file
    df.to_csv(matching_out_csv, index=False)

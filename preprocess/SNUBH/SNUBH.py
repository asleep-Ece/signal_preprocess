import os
import logging
from datetime import datetime

import librosa
import numpy as np
import pandas as pd

from soundsleep.preprocess.data_generator import BaseDataGenerator
from soundsleep.preprocess import utils
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating


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
    parser.add_argument("--all_psg_path", type=str, required=True, help="Path to the audio data")
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


class SNUBH(BaseDataGenerator):
    def __init__(self, parser):
        super(SNUBH, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

        # PSG path that has data1, data2, etc.
        self.all_psg_path = self.args.all_psg_path
        print("all_psg_path: {}".format(self.all_psg_path))

        # Output path to save npy files
        if self.args.audio_to_numpy:
            self.output_path = self.args.output_path
            os.makedirs(self.output_path, exist_ok=False)
            print("output_path: {}".format(self.output_path))

    def audio2numpy(self):
        """Preprocessing audio data and saved in numpy format."""

        logging.basicConfig(filename=os.path.join(self.output_path, "preprocess.log"), level=logging.CRITICAL)

        # Dataframe to store sampling rate information
        df_samplingRate = pd.DataFrame(columns=["Data", "ID", "Audio", "SR", "Zero portion"])
        out_csv = os.path.join(self.output_path, "sampling_rate.csv")

        for data_dir in os.listdir(self.all_psg_path):
            # Each data directory must have a `complete` folder
            sub_psg_path = os.path.join(self.all_psg_path, data_dir, "complete")
            if not os.path.isdir(sub_psg_path):
                continue

            # Prepare patient list in the current data_dir
            patient_list = [os.path.join(sub_psg_path, x) for x in os.listdir(sub_psg_path)]
            patient_list = [x for x in patient_list if os.path.isdir(x)]
            if len(patient_list) == 0:
                continue

            # Output path for current data_dir
            sub_out_path = os.path.join(self.output_path, data_dir)

            print(f"\nData extraction for {sub_psg_path} (number of subjects is {len(patient_list)})...")
            df = extract_data(patient_list, sub_out_path, self.args)
            df_samplingRate = pd.concat((df_samplingRate, df), ignore_index=True)

            # Sampling rate CSV file
            df_samplingRate.to_csv(out_csv, index=False)

        print("\nData saved in {}.".format(self.output_path))


def extract_data(patient_list, output_path, args):
    """Get audio and label data, then save them into output_path in npy format."""

    os.makedirs(output_path, exist_ok=False)

    n_patients = len(patient_list)
    start_time = datetime.now()

    # Data frame for sampling rate
    df = pd.DataFrame(columns=["Data", "ID", "Audio", "SR", "Zero portion"])
    data_dir = os.path.basename(output_path)

    for i, patient_dir in enumerate(patient_list):
        start = datetime.now()

        try:
            data_offset = utils.calculate_data_offset(patient_dir)
            extract_label = utils.extract_label_file(patient_dir)
        except (RuntimeError, ValueError) as e:
            log = "[{}] {}".format(os.path.basename(patient_dir), e)
            print(log)
            logging.critical(log)
            continue

        if data_offset == False or extract_label == False:
            log = "[{}] data_offset = {}, extract_label = {}. Skipping.".format(
                os.path.basename(patient_dir), data_offset, extract_label
            )
            print(log)
            logging.critical(log)
            continue

        # Extracting audio assuming that WAV files are already extracted
        audio_list = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir) if x.endswith(".wav")]
        label_file = os.path.join(patient_dir, "sleep_labels.csv")
        offset_file = os.path.join(patient_dir, "data_offset.csv")
        try:
            LABEL_START = pd.read_csv(offset_file)["label_start"].values[0]
        except (RuntimeError, IndexError) as e:
            log = "[{}] Cannot extract LABEL_START: {}".format(os.path.basename(patient_dir), e)
            print(log)
            logging.critical(log)
            continue

        if len(audio_list) == 0:
            log = "[{}] No audio files.".format(os.path.basename(patient_dir))
            print(log)
            logging.critical(log)
            continue

        # Load label
        labels_data = utils.read_labels(label_file)
        if len(labels_data) == 0:
            log = "[{}] No labels data.".format(os.path.basename(patient_dir))
            print(log)
            logging.critical(log)
            continue

        for audio_file in audio_list:
            audio_idx = int(os.path.basename(audio_file)[:-4].split("_")[1])

            # Get the corresponding XML file
            vid_xml_file = audio_file.split("/")
            index = os.path.splitext(vid_xml_file[-1])[0].split("_")[1]
            vid_xml_file[-1] = "video_{}.xml".format(index)
            vid_xml_file = "/".join(vid_xml_file)

            if not os.path.exists(vid_xml_file):
                log = "[{}] No corresponding video XML file {}.".format(os.path.basename(patient_dir), vid_xml_file)
                print(log)
                logging.critical(log)
                continue

            # Load audio
            audio_data, sr = librosa.load(audio_file, sr=None)  # audio_file must have WAV format

            # Skip if audio length is less than 30 seconds
            if len(audio_data) < 30 * sr:
                log = "[{}] Audio file is less than 30 seconds. Skipping.".format(audio_file)
                print(log)
                logging.critical(log)
                continue

            # Skip if zero portion is more than expected
            zero_portion = utils.get_zero_portion(audio_data)
            if zero_portion > args.upper_bound_zero_portion:
                log = "[{}] Audio data has {:.2%} zero values, which is more than the allowed threshold: {:.2%}".format(
                    audio_file, zero_portion, args.upper_bound_zero_portion
                )
                print(log)
                logging.critical(log)
                continue

            # Synchronize the audio with the label
            AUDIO_START = utils.audio_start_time(vid_xml_file)
            filename = "/".join(audio_file.split("/")[-2:])
            indices = utils.sync_audio_label(len(audio_data), sr, len(labels_data), AUDIO_START, LABEL_START, filename)
            if (type(indices) == bool) and (indices == False):
                continue
            else:
                audio_data = audio_data[indices[0] : indices[1]]
                labels = labels_data[indices[2] : indices[3]]

            # Skip if length of audio data becomes 0 after synchronization
            if len(audio_data) == 0:
                log = "[{}] Audio length becomes 0 after synchronization.".format(audio_file)
                print(log)
                logging.critical(log)
                continue

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
                file_name = os.path.join(
                    output_path, os.path.basename(patient_dir) + "_{}_{}.npy".format(audio_idx, file_idx)
                )
                np.save(file_name, data_dict)

                # Update indices
                data_idx, label_idx, file_idx = update_idx(data_idx, label_idx, file_idx)

            del audio_data

            # Sampling rate dataframe
            row_data = [
                data_dir,
                os.path.basename(patient_dir),
                os.path.basename(audio_file),
                sr,
                f"{zero_portion:.2%}",
            ]
            df = df.append(pd.Series(row_data, index=df.columns), ignore_index=True)

        if ((i + 1) % 10 == 0) or (i == 0) or (i == n_patients - 1):
            now = datetime.now()
            time = now - start
            elapsed = now - start_time
            expected = time * n_patients
            print(
                "[{}/{}] time = {}, time_elapsed = {}, time_expected = {}".format(
                    i + 1, n_patients, time, elapsed, expected
                )
            )

    return df

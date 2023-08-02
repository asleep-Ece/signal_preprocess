import os
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from soundsleep.preprocess.data_generator import BaseDataGenerator
from soundsleep.preprocess import utils
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating


def add_arguments(parser):
    parser.add_argument("--seq_len", type=int, default=20, metavar="N", help="Number of 30-sec epochs in a sequence")
    parser.add_argument("--sample_time", type=int, default=30, metavar="N", help="Time in one sample (in seconds)")
    parser.add_argument(
        "--segment_len",
        type=float,
        default=30,
        metavar="N",
        help="Time of each segment (in seconds) in which the noise is estimated",
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
    return parser


class SequentialSNUBH(BaseDataGenerator):
    def __init__(self, parser):
        super(SequentialSNUBH, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

        # Audio path
        self.audio_path = self.args.audio_path
        print("audio_path: {}".format(self.audio_path))

        # Output path to save npy files
        if self.args.audio_to_numpy:
            self.output_path = self.args.output_path
            os.makedirs(self.output_path, exist_ok=False)
            print("output_path: {}".format(self.output_path))

    def audio2numpy(self):
        """Preprocessing audio data and saved in numpy format."""

        logging.basicConfig(filename=os.path.join(self.output_path, "preprocess.log"), level=logging.CRITICAL)

        print("Dividing audio list into training, validation, and test lists...")
        filter_list = None  # TODO: make this an argument?
        train_list, valid_list, test_list = utils.divide_training_data(self.audio_path, self.args.portion, filter_list)
        print("Train: {}, Valid: {}, Test: {}".format(len(train_list), len(valid_list), len(test_list)))

        print("\nData extraction for training set...")
        train_path = os.path.join(self.output_path, "train")
        extract_data(train_list, train_path, self.args)

        print("\nData extraction for validation set...")
        valid_path = os.path.join(self.output_path, "valid")
        extract_data(valid_list, valid_path, self.args)

        print("\nData extraction for test set...")
        test_path = os.path.join(self.output_path, "test")
        extract_data(test_list, test_path, self.args)

        print("\nData saved in {}.".format(self.output_path))


def extract_data(patient_list, output_path, args):
    """Get audio and label data, then save them into output_path in npy format."""

    os.makedirs(output_path, exist_ok=False)

    n_patients = len(patient_list)
    start_time = datetime.now()

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

        audio_list = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir) if ".wma" in x]
        label_file = os.path.join(patient_dir, "sleep_labels.csv")
        offset_file = os.path.join(patient_dir, "data_offset.csv")
        LABEL_START = pd.read_csv(offset_file)["label_start"].values[0]

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

        for k, audio_file in enumerate(audio_list):
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
            audio, sr = utils.load_audio(audio_file)
            if (type(audio) == bool) and (audio == False):
                continue

            # Synchronize the audio with the label
            AUDIO_START = utils.audio_start_time(vid_xml_file)
            filename = "/".join(audio_file.split("/")[-2:])
            indices = utils.sync_audio_label(len(audio), sr, len(labels_data), AUDIO_START, LABEL_START, filename)
            if (type(indices) == bool) and (indices == False):
                continue
            else:
                audio = audio[indices[0] : indices[1]]
                labels = labels_data[indices[2] : indices[3]]

            # Noise reduction
            gated_audio = adaptive_noise_reduce(
                audio,
                sr,
                args.segment_len,
                estimate_noise_method=noise_minimum_energy,
                reduce_noise_method=spectral_gating,
                smoothing=args.smoothing,
            )
            del audio

            # Prepare important constants for data extraction
            SAMPLE_LEN = int(args.sample_time * sr)
            DATA_STRIDE = int(30 * sr)
            SEQ_LEN = args.seq_len
            LSTM_SAMPLE_LEN = (SEQ_LEN - 1) * DATA_STRIDE + SAMPLE_LEN
            LSTM_DATA_STRIDE = SEQ_LEN * DATA_STRIDE
            LSTM_LABEL_STRIDE = SEQ_LEN
            N_HEAD_TAIL = int((args.sample_time - 30) / 2 / 30)

            def update_idx(data_idx, label_idx, file_idx):
                data_idx += LSTM_DATA_STRIDE
                label_idx += LSTM_LABEL_STRIDE
                file_idx += 1
                return data_idx, label_idx, file_idx

            # Read and extract data
            data_idx = 0
            label_idx = N_HEAD_TAIL
            file_idx = 0
            data_len = len(gated_audio)

            while True:
                # Break condition
                if (data_idx + LSTM_SAMPLE_LEN) > data_len:
                    assert (data_idx / (30 * sr) + N_HEAD_TAIL) == label_idx
                    break

                # Get labels
                y_5c = labels[label_idx : label_idx + LSTM_LABEL_STRIDE]
                y_2c, y_3c, y_4c = utils.convert_stage_list(y_5c)

                # Get data
                data = gated_audio[data_idx : data_idx + LSTM_SAMPLE_LEN]
                melspec_sequence_data = utils.mel_spec_sequence(data, sr, SAMPLE_LEN, DATA_STRIDE, args)
                assert melspec_sequence_data.shape[0] == SEQ_LEN
                assert melspec_sequence_data.shape[1] == args.n_mels

                # Save data into a npy file in the dictionary format
                data_dict = {"x": melspec_sequence_data, "2c": y_2c, "3c": y_3c, "4c": y_4c, "5c": y_5c}
                file_name = os.path.join(output_path, os.path.basename(patient_dir) + "_{}_{}.npy".format(k, file_idx))
                np.save(file_name, data_dict)

                # Update indices
                data_idx, label_idx, file_idx = update_idx(data_idx, label_idx, file_idx)

            del gated_audio

        if ((i + 1) % 10 == 0) or (i == 0) or (i == n_patients - 1):
            now = datetime.now()
            time = now - start
            elapsed = now - start_time
            expected = time * n_patients
            print(
                "[{}/{}] num_samples = {}, time = {}, time_elapsed = {}, time_expected = {}".format(
                    i + 1, n_patients, file_idx + 1, time, elapsed, expected
                )
            )

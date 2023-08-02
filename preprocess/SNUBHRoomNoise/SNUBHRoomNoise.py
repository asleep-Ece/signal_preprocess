import os
import argparse
import signal
import logging
import subprocess
from datetime import datetime

import random
import librosa
import numpy as np
import pandas as pd
from natsort import natsorted

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
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio data")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the preprocessed data")
    parser.add_argument("--noise_path", type=str, required=True, help="Path to noise data")
    parser.add_argument(
        "-nr_before",
        "--noise_reduction_before",
        action="store_true",
        help="Add noise reduction before adding room noise",
    )
    parser.add_argument(
        "-nr_after", "--noise_reduction_after", action="store_true", help="Add noise reduction after adding room noise"
    )
    parser.add_argument(
        "--noise_selection_type",
        type=str,
        required=True,
        choices=["random", "uniform"],
        help="Selection type for picking room noise",
    )
    parser.add_argument(
        "--num_noise",
        type=int,
        required=True,
        help="Number of noise data to be included. \
                             Set to 0 to use all of the noise data.",
    )
    parser.add_argument(
        "--snr_selection_type",
        type=str,
        required=True,
        choices=["random", "uniform"],
        help="Selection type for picking SNR value",
    )
    parser.add_argument(
        "--snr_list",
        nargs="+",
        required=True,
        default=["5", "10", "15", "20"],
        help="SNR values for speech and noise mixing",
    )
    return parser


class SNUBHRoomNoise(BaseDataGenerator):
    def __init__(self, parser):
        super(SNUBHRoomNoise, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

    def audio2numpy(self):
        """Preprocessing audio data and saved in numpy format."""

        print("\nObtaining all patient files in test data directory")
        patient_list = []
        for data_folder in os.listdir(self.args.audio_path):
            data_path = os.path.join(self.args.audio_path, data_folder)
            for patient_folder in natsorted(os.listdir(data_path)):
                patient_path = os.path.join(data_path, patient_folder)
                patient_list.append(patient_path)
        patient_list = [os.path.join(self.args.audio_path, x) for x in patient_list]

        # Label folder name based on noise selection type and SNR
        # test-%s : signal dataset name
        # noise-%s : noise dataset name
        # nrb-%i : noise reduce before on/off
        # nra-%i : noise reduce after on/off
        # ntype-%s : noise selection type
        # cnt-%i : number of noise data
        # stype-%s : snr selection type
        # snr-%s : snr list
        test_dataset_name = self.args.audio_path.split("/")[-1]
        noise_dataset_name = self.args.noise_path.split("/")[-1]
        snr_name = "-".join(self.args.snr_list)
        folder_name = "test-%s_noise-%s_nrb-%i_nra-%i_ntype-%s_cnt-%i_stype-%s_snr-%s" % (
            test_dataset_name,
            noise_dataset_name,
            int(self.args.noise_reduction_before),
            int(self.args.noise_reduction_after),
            self.args.noise_selection_type,
            self.args.num_noise,
            self.args.snr_selection_type,
            snr_name,
        )

        output_path = os.path.join(self.args.output_path, folder_name)
        print("output_path: {}".format(output_path))

        extract_data(patient_list, output_path, self.args)
        print("\nData saved in {}.".format(output_path))

        logging.basicConfig(filename=os.path.join(output_path, "preprocess.log"), level=logging.CRITICAL)


def extract_data(patient_list, output_path, args):
    """Get audio and label data, then save them into output_path in npy format."""

    n_patients = len(patient_list)
    start_time = datetime.now()

    # Get parameters for adding room noise
    room_environment_noise_list = [
        os.path.join(args.noise_path, filename) for filename in sorted(os.listdir(args.noise_path))
    ]

    if args.num_noise < 0:
        raise ValueError(f"num_noise = {args.num_noise}. num_noise must be 0 or positive.")
    elif args.num_noise > 0:
        room_environment_noise_list = room_environment_noise_list[: args.num_noise]
    else:  # Use all noise types
        pass

    snr_list = [int(x) for x in args.snr_list]

    os.makedirs(output_path, exist_ok=False)

    output_data_path = os.path.join(output_path, "test")
    os.makedirs(output_data_path, exist_ok=False)

    out_file = open(os.path.join(output_path, "logs_signal-noise-snr.csv"), "w+")
    print("patient_id,noise_id,snr_id", file=out_file)

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

        audio_list = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir) if ".wav" in x]
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

        # Select room environment noise for 1 patient
        if args.noise_selection_type == "random":
            selected_room_noise = random.choice(room_environment_noise_list)
        elif args.noise_selection_type == "uniform":
            current_noise_idx = i % len(room_environment_noise_list)
            selected_room_noise = room_environment_noise_list[current_noise_idx]
        else:
            raise ValueError("Unexpected noise selection type: {}".format(args.noise_selection_type))

        if args.snr_selection_type == "random":
            selected_snr = random.choice(snr_list)
        elif args.snr_selection_type == "uniform":
            current_snr_idx = i % len(snr_list)
            selected_snr = snr_list[current_snr_idx]
        else:
            raise ValueError("Unexpected SNR selection type: {}".format(args.snr_selection_type))

        noise_data, noise_sr = librosa.load(selected_room_noise, sr=None)

        patient_id = patient_dir.split("/")[-1]
        noise_id = selected_room_noise.split("/")[-1].split(".")[0]
        snr_id = str(selected_snr)
        print("%s,%s,%s" % (patient_id, noise_id, snr_id), file=out_file)

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
            audio_data, audio_sr = librosa.load(audio_file, sr=None)

            # Check if sound and noise data have the same sampling rate
            assert audio_sr == noise_sr, f"audio_sr {audio_sr} is not matched to noise_sr {noise_sr}"

            # Check if audio length is less than 30 seconds
            if len(audio_data) < 30 * audio_sr:
                log = "[{}] {} is less than 30 seconds. Length is {}".format(
                    os.path.basename(patient_dir), "/".join(audio_file.split("/")[-3:]), len(audio_data)
                )
                print(log)
                logging.critical(log)
                continue

            # Synchronize the audio with the label
            AUDIO_START = utils.audio_start_time(vid_xml_file)
            filename = "/".join(audio_file.split("/")[-2:])

            indices = utils.sync_audio_label(
                len(audio_data), audio_sr, len(labels_data), AUDIO_START, LABEL_START, filename
            )

            if (type(indices) == bool) and (indices == False):
                continue
            else:
                audio_data = audio_data[indices[0] : indices[1]]
                labels = labels_data[indices[2] : indices[3]]

            # Check size of audio data after synchronizing with labels
            if len(audio_data) == 0:
                log = "[{}] {} is empty.".format(os.path.basename(patient_dir), "/".join(audio_file.split("/")[-3:]))
                print(log)
                logging.critical(log)
                continue

            # Add adaptive_noise_reduce before adding room noise signal
            if args.noise_reduction_before:
                audio_data = adaptive_noise_reduce(
                    audio_data,
                    audio_sr,
                    args.segment_len,
                    estimate_noise_method=noise_minimum_energy,
                    reduce_noise_method=spectral_gating,
                    smoothing=args.smoothing,
                )

            # Add room environment noise which is equal to audio data length
            noise_multiplier = (len(audio_data) // len(noise_data)) + 1
            noise_data = np.tile(noise_data, noise_multiplier)
            noise_data = noise_data[: len(audio_data)]

            noisy_audio_data = utils.add_noise_to_signal(audio_data, noise_data, selected_snr)

            # Add adaptive_noise_reduce after adding room noise signal
            if args.noise_reduction_after:
                noisy_audio_data = adaptive_noise_reduce(
                    noisy_audio_data,
                    audio_sr,
                    args.segment_len,
                    estimate_noise_method=noise_minimum_energy,
                    reduce_noise_method=spectral_gating,
                    smoothing=args.smoothing,
                )

            # Prepare important constants for data extraction
            SAMPLE_LEN = int(args.sample_time * audio_sr)
            DATA_STRIDE = int(30 * audio_sr)
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
            data_len = len(noisy_audio_data)

            while True:
                # Break condition
                if (data_idx + SAMPLE_LEN) > data_len:
                    assert (data_idx / (30 * audio_sr) + N_HEAD_TAIL) == label_idx
                    break

                # Get labels
                y_5c = labels[label_idx]
                y_2c, y_3c, y_4c = utils.convert_stage(y_5c)

                # Extract data
                audio_30sec = noisy_audio_data[data_idx : data_idx + SAMPLE_LEN]

                # Mel Spectrogram
                mel_spec = utils.mel_spectrogram(audio_30sec, audio_sr, args.n_mels, args.window_size, args.stride)

                # Save into a npy file in the dictionary format
                data_dict = {"x": mel_spec, "2c": y_2c, "3c": y_3c, "4c": y_4c, "5c": y_5c}
                file_name = os.path.join(
                    output_data_path, os.path.basename(patient_dir) + "_{}_{}.npy".format(k, file_idx)
                )
                np.save(file_name, data_dict)

                # Update indices
                data_idx, label_idx, file_idx = update_idx(data_idx, label_idx, file_idx)

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

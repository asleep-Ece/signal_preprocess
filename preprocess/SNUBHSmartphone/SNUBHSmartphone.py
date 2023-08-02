import os
import argparse
import signal
import logging
import subprocess
import re
import datetime

import numpy as np
import pandas as pd
import librosa

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
    parser.add_argument("--matching_file", type=str, required=True, help="CSV file containing matching data")
    parser.add_argument("-nroff", "--noise_reduce_off", action="store_true", help="Turn off noise reduction")
    return parser


class SNUBHSmartphone(BaseDataGenerator):
    def __init__(self, parser):
        super(SNUBHSmartphone, self).__init__()

        # Parse arguments
        parser = add_arguments(parser)
        self.args = parser.parse_args()

        # Excel files
        matching_file = self.args.matching_file
        self.matching_data = pd.read_csv(matching_file).to_numpy()

        # PSG paths
        self.psg_paths = {
            "A": "/mnt/assd2/complete",
            "B": "/mnt/assd1/complete",
            "C": "/mnt/assd3/complete",
            "D": "/mnt/assd4/Data4",
        }

        self.ssd_dirs = {"A": "data2", "B": "data1", "C": "data3", "D": "data4"}

        # Smartphone audio path
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
        extract_data(
            train_list, self.output_path, "train", self.args, self.psg_paths, self.matching_data, self.ssd_dirs
        )

        print("\nData extraction for validation set...")
        extract_data(
            valid_list, self.output_path, "valid", self.args, self.psg_paths, self.matching_data, self.ssd_dirs
        )

        print("\nData extraction for test set...")
        extract_data(test_list, self.output_path, "test", self.args, self.psg_paths, self.matching_data, self.ssd_dirs)

        print("\nData saved in {}.".format(self.output_path))


def join_psg_path(psg_paths, data_code):
    """data_code has format A-XX or B-XX or C-XX."""
    key, idx = data_code.split("-")
    matched_psg = "{:03d}_data".format(int(idx))
    matched_psg = os.path.join(psg_paths[key], matched_psg)
    return matched_psg


def find_matched_psg(audio_file, psg_paths, matching_data):
    """Given audio_file and matching_data, find the corresponding PSG data path in psg_paths."""
    matched_psg = None
    time_offset = None
    xml_id = None
    ssd_key = None

    filename = os.path.basename(audio_file)
    if len(filename.split(" ")) == 3:
        ID = int(filename.split(" ")[1])
        audio_date = os.path.splitext(filename.split(" ")[-1])[0]
        audio_date = "20" + re.sub("[(.)]", "", audio_date)  # 20yymmdd
        audio_date = datetime.datetime.strptime(audio_date, "%Y%m%d").strftime("%Y-%m-%d")  # 20yy-mm-dd
    elif len(filename.split(" ")) == 2:
        ID = int(filename.split(" ")[-1].split(".")[0])
        audio_date = None
    else:
        log = "Unexpected case: filename = {}".format(filename)
        logging.critical("Unexpected case: filename = {}".format(filename))
        print(log)
        return matched_psg, time_offset, xml_id, ssd_key

    # Match ID
    row_data = matching_data[matching_data[:, 1] == ID]
    if len(row_data) == 0:
        log = "[{}] No matched ID.".format(filename)
        print(log)
        logging.critical(log)
        return matched_psg, time_offset, xml_id, ssd_key

    for i, row in enumerate(row_data):
        # Check data code
        data_code = row[0]
        if not isinstance(data_code, str) and np.isnan(data_code):
            log = "[{}] No matched PSG data. (i={}/{})".format(filename, i, len(row_data))
            print(log)
            logging.critical(log)
            continue

        # Check sync data
        time_offset = row[3]
        xml_id = row[4]
        if (not isinstance(time_offset, str) and np.isnan(time_offset)) or (
            not isinstance(xml_id, str) and np.isnan(xml_id)
        ):
            time_offset = xml_id = None  # Change to None for easier coding
            log = "[{}] Not enough information for synchronization: time_offset = {} and xml_id = {}".format(
                filename, time_offset, xml_id
            )
            print(log)
            logging.critical(log)
            continue

        psg_date = row[2]
        if audio_date != None and isinstance(psg_date, str):
            if psg_date == audio_date:
                matched_psg = join_psg_path(psg_paths, data_code)
                ssd_key = data_code[0]
                break
            else:
                log = "[{}] Date not matched: audio_date = {}, psg_date = {} (i={}/{})".format(
                    filename, audio_date, psg_date, i, len(row_data)
                )
                print(log)
                logging.critical(log)
                continue
        else:  # Cannot verify the date of taking sleep analysis
            if len(row_data) > 1:
                log = "[{}] There's more than one matched PSG data without enough date information. Potential wrong match. (i={}/{})".format(
                    filename, i, len(row_data)
                )
                print(log)
                logging.critical(log)
                continue
            matched_psg = join_psg_path(psg_paths, data_code)
            ssd_key = data_code[0]

    return matched_psg, time_offset, xml_id, ssd_key


def get_smp_audio_start_time(xml_video_start, time_offset):
    # PSG video time
    hh, mm, ss = xml_video_start.split(":")
    hh, mm, ss = int(hh), int(mm), int(ss)
    origin = datetime.datetime(year=2021, month=5, day=14, hour=hh, minute=mm, second=ss)

    # Time offset
    if "-" in time_offset:
        act = "sub"
        assert len(time_offset) == 9
        hh, mm, ss = time_offset[1:].split(":")
    else:
        act = "add"
        assert len(time_offset) == 8
        hh, mm, ss = time_offset.split(":")
    hh, mm, ss = int(hh), int(mm), int(ss)
    offset = datetime.timedelta(hours=hh, minutes=mm, seconds=ss)

    # Start time of smartphone audio
    if act == "sub":
        res = (origin - offset).strftime("%H:%M:%S")
    else:  # act == 'add'
        res = (origin + offset).strftime("%H:%M:%S")

    return res


def extract_data(audio_list, output_path, mode, args, psg_paths, matching_data, ssd_dirs):
    """Get audio and label data, then save them into output_path in npy format."""

    n_files = len(audio_list)
    start_time = datetime.datetime.now()

    for i, audio_file in enumerate(audio_list):
        start = datetime.datetime.now()

        # Find the corresponding matched_psg
        matched_psg, time_offset, xml_id, ssd_key = find_matched_psg(audio_file, psg_paths, matching_data)
        if matched_psg == None or time_offset == None or xml_id == None:
            continue
        if not os.path.exists(matched_psg):
            log = "[{}] {} is matched but does not exist.".format(os.path.basename(audio_file), matched_psg)
            print(log)
            logging.critical(log)
            continue

        try:
            data_offset = utils.calculate_data_offset(matched_psg)
            extract_label = utils.extract_label_file(matched_psg)
        except (RuntimeError, ValueError) as e:
            log = "[{}] {}".format(os.path.basename(matched_psg), e)
            print(log)
            logging.critical(log)
            continue

        if data_offset == False or extract_label == False:
            log = "[{}] {}: data_offset = {}, extract_label = {}. Skipping.".format(
                os.path.basename(audio_file), matched_psg, data_offset, extract_label
            )
            print(log)
            logging.critical(log)
            continue

        label_file = os.path.join(matched_psg, "sleep_labels.csv")
        offset_file = os.path.join(matched_psg, "data_offset.csv")
        LABEL_START = pd.read_csv(offset_file)["label_start"].values[0]

        # Load label
        labels_data = utils.read_labels(label_file)
        if len(labels_data) == 0:
            log = "[{}] {}: No labels data.".format(os.path.basename(audio_file), matched_psg)
            print(log)
            logging.critical(log)
            continue

        # Load audio
        out_file = "16kHz_audio.wav"
        try:
            # Convert to 16 kHz and load
            subprocess.run(
                ["ffmpeg", "-i", audio_file, "-ar", "16000", out_file],
                timeout=90,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            log = "[ffmpeg] Conversion error happened when converting {} to 16 kHz version: {}.".format(audio_file, e)
            print(log)
            logging.critical(log)
            if os.path.exists(out_file):
                os.remove(out_file)
            continue
        audio_data, sr = librosa.load(out_file, sr=None)
        os.remove(out_file)

        # Synchronize the audio with the label
        vid_xml_file = os.path.join(matched_psg, "video_{}.xml".format(int(xml_id)))
        XML_VIDEO_START = utils.audio_start_time(vid_xml_file)
        AUDIO_START = get_smp_audio_start_time(XML_VIDEO_START, time_offset)
        filename = os.path.basename(audio_file)
        indices = utils.sync_audio_label(len(audio_data), sr, len(labels_data), AUDIO_START, LABEL_START, filename)
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

        # Saving directory
        save_dir = os.path.join(output_path, ssd_dirs[ssd_key], mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

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
            file_name = os.path.join(save_dir, "{}_0_{}.npy".format(os.path.basename(matched_psg), file_idx))
            np.save(file_name, data_dict)

            # Update indices
            data_idx, label_idx, file_idx = update_idx(data_idx, label_idx, file_idx)

        del audio_data

        if ((i + 1) % 10 == 0) or (i == 0) or (i == n_files - 1):
            now = datetime.datetime.now()
            time = now - start
            elapsed = now - start_time
            expected = time * n_files
            print(
                "[{}/{}] num_samples = {}, time = {}, time_elapsed = {}, time_expected = {}".format(
                    i + 1, n_files, file_idx + 1, time, elapsed, expected
                )
            )

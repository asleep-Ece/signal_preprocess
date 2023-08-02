import os
import logging
from datetime import datetime, timedelta
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import librosa
import numpy as np
from tqdm import tqdm

from soundsleep.preprocess.data_generator import BaseDataGenerator
from soundsleep.preprocess import utils
from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating


def add_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--sample_time", type=int, default=30, metavar="N", help="Time in one sample (in seconds)")
    parser.add_argument(
        "--past_future_time", type=int, default=0, metavar="N", help="Time to look back and ahead (in seconds)"
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=30,
        metavar="N",
        help="Window stride which determines the stride between two consecutive data and event labels",
    )
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
    return parser


class SNUBHEventDetection(BaseDataGenerator):
    def __init__(self, parser: ArgumentParser) -> None:
        super(SNUBHEventDetection, self).__init__()

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

    def audio2numpy(self) -> None:
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


def update_idx(
    data_idx: int, window_start_time_stamp: int, file_idx: int, data_stride: int, window_stride: int
) -> Tuple[int, int, int, int]:
    """Move audio segment, label id, audio window, and output file id by one stride."""
    data_idx += data_stride
    window_start_time_stamp += window_stride
    file_idx += 1
    return data_idx, window_start_time_stamp, file_idx


def date_time_converter(time_string: str) -> datetime:
    """Convert time string to DateTime object."""
    return datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")


def fill_frame_with_event_label(
    patient_id: str, audio_start: str, audio_duration: float, events: List[Dict[str, str]]
) -> List[int]:
    """Returns an event frame representing all events during sleep.

    Returns a frame representing all events during one whole night.
    The frame contains multiple segments, each segment represents
    an event happening in one second.

    1. Initially sets all segments of the frame to 0, meaning no event.
    2. For each event in the event list, extracts event type, start time,
       and end time, then fills the corresponding segments with:

        0: NO-EVENT
        1: APNEA
        2: HYPOPNEA

    Args:
        patient_id: ID of the patient.
        audio_start: Start timestamp of the audio segment.
        audio_duration: Audio duration in seconds.
        events: List of events happen in the audio segment.

    Returns:
        A frame with segments filled with event labels.
    """
    audio_start_second_since_epoch = date_time_converter(audio_start).timestamp()
    audio_end = date_time_converter(audio_start) + timedelta(seconds=audio_duration)
    audio_stop_second_since_epoch = audio_end.timestamp()

    frame = [0] * int(audio_duration)
    event_type_dict = {"APNEA": 1, "HYPOPNEA": 2}

    for event in events:
        event_type, event_start, event_stop = event["event_type"], event["start_time"], event["stop_time"]
        event_start_second_since_epoch = date_time_converter(event_start).timestamp()
        event_stop_second_since_epoch = date_time_converter(event_stop).timestamp()

        # Safety check if the timestamp of the event is out of range
        if event_start_second_since_epoch < audio_start_second_since_epoch:
            event_start_second_since_epoch = audio_start_second_since_epoch
        if event_stop_second_since_epoch > audio_stop_second_since_epoch:
            event_stop_second_since_epoch = audio_stop_second_since_epoch

        # Calculate the start and stop index of the event compare to the audio time
        event_start_since_audio_start = int(event_start_second_since_epoch - audio_start_second_since_epoch)
        event_stop_since_audio_start = int(event_stop_second_since_epoch - audio_start_second_since_epoch)
        for idx in range(
            event_start_since_audio_start, event_stop_since_audio_start + 1
        ):  # +1 to include the stop index
            # Check for confliction, for now we will just replace the previous value if happen
            if frame[idx] != 0:
                log = f"""[CONFLICT {patient_id}] Attempting to fill with {event_type_dict[event_type]}
                          at {datetime.fromtimestamp(audio_start_second_since_epoch + idx).strftime("%Y-%m-%dT%H:%M:%S.%f")},
                          but already filled with {frame[idx]}"""
                print(log)
                logging.warning(log)

            # Fill the frame with the event label
            frame[idx] = event_type_dict[event_type]
            assert event_type in event_type_dict.keys(), f"Unknown event type of {event_type}"
    return frame


def get_label_for_time_window(
    frame: List[int], start_time_stamp: int, end_time_stamp: int, threshold: int = 0, merge_apnea_hypopnea: bool = True
) -> int:
    """Get the label for the time window.

    We count the number of items in the window that each event happens. The event for the whole window
    is the dominated event except for NO-EVENT if the number of seconds it appears in the window is
    higher than threshold. Labels are:
    if merge_apnea_hypopnea:
        - 0: NO-EVENT
        - 1: APNEA/HYPOPNEA
    if not:
        - 0: NO-EVENT
        - 1: APNEA
        - 2: HYPOPNEA

    Args:
        frame: frame that was filled with events.
        start_time_stamp: start time (seconds since audio start) of the window we want to get the label.
        end_time_stamp: end time (seconds since audio start) of the window we want to get the label.
        threshold: threshold to decide if the event should be counted.
        merge_apnea_hypopnea: whether to merge apnea and hypopnea into one event.
    Returns:
        The label for the window.
    """
    window = np.array(frame[start_time_stamp:end_time_stamp])
    # Change labels of Hyponea if merge_apnea_hypopnea, do nothing otherwise
    if merge_apnea_hypopnea:
        window = np.where(window == 2, 1, window)

    count_each_event = np.bincount(window)
    # Set as no event if the events are not long enough
    if max(count_each_event) < threshold:
        return 0

    return np.argmax(count_each_event).item()


def extract_data(patient_list: List[str], output_path: str, args: Namespace) -> None:
    """Get audio and label data, then save them into output_path in npy format."""

    os.makedirs(output_path, exist_ok=False)

    for patient_dir in tqdm(patient_list, desc="Extracting data"):
        audio_list = [os.path.join(patient_dir, x) for x in os.listdir(patient_dir) if ".wav" in x]

        if len(audio_list) == 0:
            log = "[{}] No audio files.".format(os.path.basename(patient_dir))
            print(log)
            logging.critical(log)
            continue

        events = utils.extract_event_segments(patient_dir)

        # Check for empty event list
        if len(events) == 0:
            log = f"[{os.path.basename(patient_dir)}] No events detected!"
            print(log)
            logging.critical(log)
            continue

        for audio_file_id, audio_file in enumerate(audio_list):
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

            # Load audio data using native sampling rate
            try:
                audio_data, sr = librosa.load(audio_file, sr=None)
            except Exception as e:
                log = "[{}] Failed to load audio file {}.\n{}".format(os.path.basename(patient_dir), audio_file, str(e))
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

            audio_start_time = utils.audio_start_datetime(vid_xml_file)
            audio_duration = len(audio_data) / sr

            patient_id = f"{os.path.basename(patient_dir)}_{audio_file_id}"
            frame_with_event_label = fill_frame_with_event_label(patient_id, audio_start_time, audio_duration, events)

            # Prepare data strides and lengths
            sample_len = int(args.sample_time * sr)  # Length of each sample
            past_future_time = args.past_future_time  # Past future time
            window_len = (
                args.sample_time - 2 * past_future_time
            )  # Duration of one window to get dominated label from event frame
            window_stride = args.window_stride  # Length of one window to get label from the event frame
            data_stride = int(window_stride * sr)  # Audio data stride

            # Read and extract data
            data_idx = 0
            window_start_time_stamp = past_future_time  # Label window starts from past_future_time
            file_idx = 0
            data_len = len(audio_data)

            while True:
                # Break condition
                if (data_idx + sample_len) > data_len:
                    break

                # Extract data for every sample_time + looking back and ahead seconds
                audio_sample = audio_data[data_idx : data_idx + sample_len]

                # Window stop time stamp to get dominated event from frame
                window_stop_time_stamp = window_start_time_stamp + window_len

                # Mel Spectrogram
                mel_spec = utils.mel_spectrogram(audio_sample, sr, args.n_mels, args.window_size, args.stride)

                # Extract the label for the time window
                y_2c = get_label_for_time_window(
                    frame_with_event_label,
                    window_start_time_stamp,
                    window_stop_time_stamp,
                    threshold=0,
                    merge_apnea_hypopnea=True,
                )
                y_3c = get_label_for_time_window(
                    frame_with_event_label,
                    window_start_time_stamp,
                    window_stop_time_stamp,
                    threshold=0,
                    merge_apnea_hypopnea=False,
                )

                # Save into a npy file in the dictionary format
                data_dict = {"x": mel_spec, "2c": y_2c, "3c": y_3c}
                file_name = os.path.join(
                    output_path, os.path.basename(patient_dir) + "_{}_{}.npy".format(audio_file_id, file_idx)
                )
                np.save(file_name, data_dict)

                # Update indices
                data_idx, window_start_time_stamp, file_idx = update_idx(
                    data_idx, window_start_time_stamp, file_idx, data_stride, window_stride
                )
            del audio_data

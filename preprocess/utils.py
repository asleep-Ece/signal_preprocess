import logging
import math
import os
import subprocess
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile


def load_audio(audio_file, sr=None):
    """Load input audio files."""
    # Get current path
    current_path = os.getcwd()

    # Convert
    audio_name = os.path.basename(audio_file)
    patient_dir = os.path.dirname(audio_file)
    out_file = "converted_audio.wav"
    os.chdir(patient_dir)

    convert = wma2wav(audio_name, out_file, patient_dir)

    if convert == False:
        return False, False

    # Load converted audio data
    sig, sr = librosa.load(out_file, sr=sr)

    # Remove converted file
    os.remove(out_file)

    # Get back to original path
    os.chdir(current_path)

    return sig, sr


def wma2wav(audio_name, out_file, patient_dir):
    tool = os.path.join(os.environ["SOUNDSLEEP_ROOT"], "tools/wma2wav.exe")

    try:
        subprocess.run([tool, "-i", audio_name, "-o", out_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return True

    except RuntimeError:
        log = "[wma2wav] There was a problem converting audio file {} of patient {}".format(audio_name, patient_dir)
        print(log)
        logging.critical(log)

        return False


def ffmpeg_load_audio(audio_file, n_trials=5):
    """Load an audio file using ffmpeg library."""

    out_file = "16kHz_audio.wav"
    for i in range(n_trials):
        try:
            # Convert to 16kHz audio and load
            subprocess.run(
                ["ffmpeg", "-i", audio_file, "-ar", "16000", out_file, "-y"],
                timeout=50,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            break
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            print("[{}/{}] Error happened while using ffmpeg to load audio: {}".format(i + 1, n_trials, e))
            if os.path.exists(out_file):
                os.remove(out_file)

    # Loading converted 16kHz audio file
    try:
        audio_data, sr = librosa.load(out_file, sr=None)
        os.remove(out_file)
    except Exception as e:
        print("Librosa loading error: {}".format(e))
        raise

    return audio_data, sr


def framing_signal(signal, sr, framesize=0.04, overlap=0.02):
    framesize = int(framesize * sr)
    overlap = int(overlap * sr)
    stride = framesize - overlap
    n_frames = ((len(signal) - framesize) // stride) + 1
    frames = np.ndarray((n_frames, framesize))
    start_end = []
    for i in range(n_frames):
        start = i * stride
        end = i * stride + framesize
        if i == n_frames - 1:
            assert len(frames) - end <= framesize
        frames[i] = signal[start:end]
        start_end.append([start, end])

    return frames, start_end


def short_time_energy(frames):
    return np.sum(frames**2, axis=1)


def write_wav(signal, sr, outfile):
    wavfile.write(outfile, sr, (signal * 32768).astype(np.int16))  # save signed 16-bit WAV format


def visualize(signal_list, title_list, drawstyle="default", save=False, path=None, show=False):
    """Visualize audio signals for comparison."""
    assert len(signal_list) == len(title_list)
    N = len(title_list)
    fig, ax = plt.subplots(N, figsize=(20, N * 3), sharex=True)
    for i in range(N):
        ax[i].plot(signal_list[i], drawstyle=drawstyle)
        ax[i].set_title(title_list[i])
    plt.tight_layout()
    if save:
        path = "fig.png" if path == None else path
        plt.savefig(path, dpi=300)
    if show:
        plt.show()
    plt.close()


def mel_spectrogram(audio_data, sample_rate, n_mels=40, window_size=40e-3, stride=20e-3):
    """Convert audio data into Mel Spectrograms using librosa."""
    n_fft = int(sample_rate * window_size)
    hop_length = int(sample_rate * stride)

    audio_feature = librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=0, fmax=8000
    )

    return audio_feature


def convert_stage(stage):
    """Convert the original 5-class stage labels to 2-class, 3-class, and 4-class types."""

    # 2-class type (Wake, Sleep)
    y_2c = 1 if stage != 0 else stage

    # 3-class type (Wake, Sleep, REM)
    y_3c = stage
    y_3c = 1 if (stage == 2) or (stage == 3) else y_3c
    y_3c = 2 if stage == 4 else y_3c

    # 4-class type (Wake, Light Sleep, Deep Sleep, REM)
    y_4c = stage
    y_4c = 1 if stage == 2 else y_4c
    y_4c = 2 if stage == 3 else y_4c
    y_4c = 3 if stage == 4 else y_4c

    return [y_2c, y_3c, y_4c]


def convert_stage_list(stage_list):
    y_2c, y_3c, y_4c = [], [], []
    for stage in stage_list:
        stage_2c, stage_3c, stage_4c = convert_stage(stage)
        y_2c.append(stage_2c)
        y_3c.append(stage_3c)
        y_4c.append(stage_4c)
    return y_2c, y_3c, y_4c


def label_string_to_int(string):
    """Receive string format of label and return integer value."""

    if string == "SLEEP-S0":
        return 0
    elif string == "SLEEP-S1":
        return 1
    elif string == "SLEEP-S2":
        return 2
    elif string == "SLEEP-S3":
        return 3
    elif string == "SLEEP-REM":
        return 4
    else:
        raise ValueError("string = {}".format(string))


def calculate_data_offset(patient_dir, remlogic_offset=9):
    column_names = ["ID", "label_start", "data_start", "offset"]
    df = pd.DataFrame(columns=column_names)

    patient_ID = os.path.basename(patient_dir).split("_")[0]
    ewp_file = os.path.join(patient_dir, os.path.basename(patient_dir) + ".ewp")
    if not os.path.exists(ewp_file):
        print("EWP file {} does not exist.".format(ewp_file))
        return False

    with open(ewp_file, "rb") as f:
        analysis_flag, recording_flag = False, False
        while True:
            line = f.readline()
            if not line:
                break
            str_list = []
            for c in line:
                c = chr(c)
                if c.isalpha() or c.isnumeric() or c == " " or c == "\n" or c == ":" or c == "-" or c == ".":
                    str_list.append(c)
            str_line = "".join(str_list)

            target_pattern = "T[0-9]{2}:[0-9]{2}:[0-9]{2}"  # 'THH:MM:SS'
            p = re.compile(target_pattern)

            if "Analysis.StartTime" in str_line:
                stop_idx = str_line.index("Analysis.StopTime")
                start_idx = str_line.index("Analysis.StartTime")
                search_str = str_line[stop_idx:start_idx]
                matched = p.search(search_str)
                label_start = search_str[matched.start() + 1 : matched.end()]
                label_start_dt = datetime.strptime(label_start, "%H:%M:%S")
                analysis_flag = True

            if "Recording.StartDate" in str_line:
                stop_idx = str_line.index("Recording.StopDate")
                start_idx = str_line.index("Recording.StartDate")
                search_str = str_line[stop_idx:start_idx]
                matched = p.search(search_str)
                data_start = search_str[matched.start() + 1 : matched.end()]
                remlogic_error = timedelta(seconds=remlogic_offset)
                data_start_dt = datetime.strptime(data_start, "%H:%M:%S") + remlogic_error
                data_start = data_start_dt.strftime("%H:%M:%S")
                recording_flag = True

            if analysis_flag and recording_flag:
                time_delta = label_start_dt - data_start_dt
                offset = str(time_delta)
                new_row = [patient_ID, label_start, data_start, offset]
                df = df.append(pd.Series(new_row, index=df.columns), ignore_index=True)
                break

    out_file = os.path.join(patient_dir, "data_offset.csv")
    df.to_csv(out_file, index=False)
    return True


def extract_label_file(patient_dir):# label
    xml_file = os.path.join(patient_dir, "data-Events.xml")
    if not os.path.exists(xml_file):
        print("XML file {} does not exist.".format(xml_file))
        return False

    root = ET.parse(xml_file).getroot()
    event_tag = "./Events/Event"
    event_list = ["SLEEP-S0", "SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4", "SLEEP-REM"]
    label_list = []
    for event in root.findall(event_tag):
        label = []
        for attr in event:
            if attr.text in event_list:
                label.append(attr.text)
                continue
            break
        if label != []:
            label_list.append(label)

    label_file = os.path.join(patient_dir, "sleep_labels.csv")
    np.savetxt(label_file, np.array(label_list), fmt="%s", delimiter=",", comments="")
    return True


def extract_event_segments(patient_dir: str) -> List[Dict[str, str]]:
    """Extract event segment from data-Events.xml file in patient_dir.

    Save extracted event segments with event type, start time, and end time to
    event_list and return the list.

    Args:
        patient_dir: directory of patient data.
        file_idx: index of audio file to extract.

    Returns:
        True if success. False if the data-Events.xml file does not exist.
    """
    event_filter = ["APNEA", "HYPOPNEA"]
    xml_file = os.path.join(patient_dir, "data-Events.xml")

    root = ET.parse(xml_file).getroot()
    events = root.findall("./Events/Event")
    event_list = []

    for event in events:
        event_dict = {}
        event_type = event.find("Type").text

        # Filter out the events whose type contains any of the event labels (e.g. 'APNEA', 'HYPOPNEA')
        if any(event_label in event_type for event_label in event_filter):
            event_start_time = event.find("StartTime").text
            event_stop_time = event.find("StopTime").text

            for event_label in event_filter:
                if event_label in event_type:
                    event_dict["event_type"] = event_label
                    break

            event_dict["start_time"] = event_start_time
            event_dict["stop_time"] = event_stop_time

            event_list.append(event_dict)

    return event_list


def read_labels(label_file):
    """Receive path to the CSV label file and return a list of integer label values."""

    with open(label_file) as f:
        labels = [label_string_to_int(x.strip()) for x in f.readlines()]

    return labels


def audio_start_time(vid_xml_file):
    """Get start time of video/audio from the XML file generated by mediainfo CLI."""
    root = ET.parse(vid_xml_file).getroot()
    xmlns = "{https://mediaarea.net/mediainfo}"
    tag = "./{}media/{}track/{}extra/{}Start".format(xmlns, xmlns, xmlns, xmlns)
    start_tag = root.find(tag)
    start_time = start_tag.text.split("T")[1].split(".")[0]
    return start_time


def audio_start_datetime(vid_xml_file: str) -> str:
    """Get start time of video/audio in datetime format from the XML file generated by mediainfo CLI."""
    root = ET.parse(vid_xml_file).getroot()
    xmlns = "{https://mediaarea.net/mediainfo}"
    start_tag = "./{}media/{}track/{}extra/{}Start".format(xmlns, xmlns, xmlns, xmlns)
    start_time = root.find(start_tag).text
    return start_time


def hhmmss_to_sec(hhmmss):
    """Convert hhmmss string to the integer value of seconds."""
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def sync_audio_label(audio_len, sr, label_len, audio_start_hhmmss, label_start_hhmmss, filename):
    """Synchronizes the audio and the labels
    so that they have same start time and same duration.
    Return start and end indices for both audio and label.
    """
    label_start = hhmmss_to_sec(label_start_hhmmss)
    audio_start = hhmmss_to_sec(audio_start_hhmmss)

    # Modify start time if necessary
    NOON_TIME = 12 * 60 * 60
    ONE_DAY_TIME = 24 * 60 * 60

    if (audio_start < NOON_TIME) and (label_start < NOON_TIME):
        # Both start after midnight
        pass
    elif audio_start < NOON_TIME:
        # Audio starts after midnight, label starts before midnight
        audio_start += ONE_DAY_TIME
    elif label_start < NOON_TIME:
        # Label starts after midnight, audio starts before midnight
        label_start += ONE_DAY_TIME
    else:
        # Both start before midnight
        pass

    # Return if there is no overlapped duration
    if audio_start + audio_len / sr < label_start:
        log = "[{}] Audio ends before labeling starts: audio_start: {}, label_start: {}".format(
            filename, audio_start_hhmmss, label_start_hhmmss
        )
        logging.critical(log)
        return False
    elif audio_start > label_start + label_len * 30:
        log = "[{}] Labeling ends before audio starts: audio_start: {}, label_start: {}".format(
            filename, audio_start_hhmmss, label_start_hhmmss
        )
        logging.critical(log)
        return False
    else:
        pass

    # Get start indices
    if audio_start < label_start:
        diff = label_start - audio_start
        label_start_idx = 0
        audio_start_idx = int(diff * sr)
    elif audio_start > label_start:
        diff = audio_start - label_start
        label_start_idx = int(np.ceil(diff / 30))
        audio_start_idx = int(label_start_idx * 30 - diff) * sr
    else:
        label_start_idx = 0
        audio_start_idx = 0

    # Get end indices
    audio_time = (audio_len - audio_start_idx) / sr
    label_time = (label_len - label_start_idx) * 30
    if audio_time < label_time:
        diff = label_time - audio_time
        label_end_idx = int(label_len - np.ceil(diff / 30))
        final_label_len = label_end_idx - label_start_idx
        audio_end_idx = int(audio_start_idx + final_label_len * 30 * sr)
    else:
        label_end_idx = label_len
        audio_end_idx = int(audio_start_idx + label_time * sr)

    if (audio_end_idx - audio_start_idx) / sr != (label_end_idx - label_start_idx) * 30:
        audio_duration = (audio_end_idx - audio_start_idx) / sr
        label_duration = (label_end_idx - label_start_idx) * 30
        log = "[{}] No match! Audio duration = {} while label duration = {}".format(
            filename, audio_duration, label_duration
        )
        logging.critical(log)
        return False
    else:
        return [audio_start_idx, audio_end_idx, label_start_idx, label_end_idx]


def divide_training_data(path, portion, filter_list=None):
    """Divide files into train set, validation set, and test set."""

    # Get the full list
    if filter_list is None:
        # Add OSX support - Ignore .DS_Store files
        patient_list = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    else:
        patient_list = []
        for p in os.listdir(path):
            index = int(p.split("_")[0])
            if index in filter_list:
                patient_list.append(p)

    # Divide the files
    train_list = np.random.choice(patient_list, int(len(patient_list) * portion), replace=False)
    tmp_list = [k for k in patient_list if k not in train_list]
    valid_list = np.random.choice(tmp_list, int(len(tmp_list) * 0.5), replace=False)
    test_list = [k for k in tmp_list if k not in valid_list]

    # Get full path
    train_list = [os.path.join(path, x) for x in train_list]
    valid_list = [os.path.join(path, x) for x in valid_list]
    test_list = [os.path.join(path, x) for x in test_list]

    return train_list, valid_list, test_list


def retrieve_mobile_data_paths(path: str) -> List[str]:
    """Retrieve data paths (used for training only).

    Retrieve the paths to all the sleep sessions recorded by all the users.
    Used by classes `soundsleep.preprocess.SmartphoneUnsupervised`
    and `soundsleep.preprocess.MbrainSurvey`.

    Return:
        train_list: list of paths to all the recorded sessions.
    """

    train_list = []
    for user in os.listdir(path):
        user_dir = os.path.join(path, user)
        if not os.path.isdir(user_dir):
            continue
        for session in os.listdir(user_dir):
            session_dir = os.path.join(user_dir, session)
            if not os.path.isdir(session_dir):
                continue
            train_list.append(session_dir)

    return train_list


def mel_spec_sequence(data, sr, sample_len, stride, args):
    """Divide and convert data into sequences of Mel Spectrograms."""

    data_len = len(data)
    seq_data = []
    idx = 0
    while True:
        if (idx + sample_len) > data_len:
            assert (idx + sample_len) == data_len + stride
            break
        sample = data[idx : idx + sample_len]
        mel_spec = mel_spectrogram(sample, sr, args.n_mels, args.window_size, args.stride)
        seq_data.append(mel_spec)
        idx += stride

    return np.array(seq_data)


def add_noise_to_signal(signal: np.ndarray, noise: np.ndarray, target_snr_db: int) -> np.ndarray:
    """
    Adds noise to the signal of interest.

    Mixes the noise to the signal with the given signal-to-noise ratio.
    Adapted from https://pytorch.org/audio/main/tutorials/audio_data_augmentation_tutorial.html.

    Args:
        signal: A sequence of signal of interest with shape [time].
        noise: A sequence of noise signal with shape [time].
        target_snr_db: The desired SNR value in dB scale.

    Returns:
        The signal with room noise with shape [time].
    """
    signal_power = np.linalg.norm(signal, 2)
    noise_power = np.linalg.norm(noise, 2)
    snr = math.exp(target_snr_db / 10)  # Convert dB to linear
    scale = snr * noise_power / signal_power
    noisy_signal = (scale * signal + noise) / 2

    return noisy_signal


def get_zero_portion(audio_data: np.ndarray) -> float:
    """Return portion of zero values in `audio_data`."""
    return (audio_data == 0).mean()

from typing import Tuple, List

import numpy as np
import librosa

from soundsleep.preprocess.noise.estimate_noise import noise_minimum_energy
from soundsleep.preprocess.noise.reduce_noise import adaptive_noise_reduce
from soundsleep.preprocess.noise.spectral_gating import spectral_gating
from soundsleep.preprocess import utils


def sam2sam_preprocess(audio_data, sr, args):
    SAMPLE_LEN = int(args.sample_time * sr)
    DATA_STRIDE = int(30 * sr)

    # Read data
    data_len = len(audio_data)
    data_idx = 0
    mel_spec_data = []

    while True:
        # Break condition
        if (data_idx + SAMPLE_LEN) > data_len:
            break

        # Extract 30-second epoch
        audio_30s = audio_data[data_idx : data_idx + SAMPLE_LEN]

        # Mel Spectrogram
        x = utils.mel_spectrogram(audio_30s, sr, args.n_mels, args.window_size, args.stride)

        # Append
        mel_spec_data.append(x)

        data_idx += DATA_STRIDE

    return mel_spec_data


def seq2seq_preprocess(audio_data, sr, args):
    SAMPLE_LEN = int(args.sample_time * sr)
    DATA_STRIDE = int(30 * sr)
    SEQ_LEN = args.n_mid + 2 * args.n_headtail
    LSTM_SAMPLE_LEN = (SEQ_LEN - 1) * DATA_STRIDE + SAMPLE_LEN
    if hasattr(args, "lstm_stride"):
        LSTM_DATA_STRIDE = args.lstm_stride * DATA_STRIDE
    else:
        LSTM_DATA_STRIDE = args.n_mid * DATA_STRIDE

    # Read and extract data
    data_idx = 0
    data_len = len(audio_data)
    mel_spec_data = []

    while True:
        # Break condition
        if (data_idx + LSTM_SAMPLE_LEN) > data_len:
            break

        # Get data
        data = audio_data[data_idx : data_idx + LSTM_SAMPLE_LEN]

        # Mel Spectrogram
        melspec_sequence_data = utils.mel_spec_sequence(data, sr, SAMPLE_LEN, DATA_STRIDE, args)
        assert melspec_sequence_data.shape[0] == SEQ_LEN
        assert melspec_sequence_data.shape[1] == args.n_mels

        # Append
        mel_spec_data.append(melspec_sequence_data)

        # Update indices
        data_idx += LSTM_DATA_STRIDE

    return mel_spec_data


def round_data(audio_data, length=5 * 60 * 16000):
    data_len = len(audio_data)
    if data_len > length:
        # Cut off redundant data
        audio_data = audio_data[-length:]
    elif data_len < length:
        # Using current data to fill in the missing parts
        diff = length - data_len
        synthetic_data = audio_data[-diff:]
        audio_data = np.concatenate((audio_data, synthetic_data))
    else:
        return audio_data

    assert len(audio_data) == length
    return audio_data


def load_audio_list(audio_files: List, ffmpeg: bool = True) -> Tuple:
    """Loads audio files in the list, assuming each audio is 5-min long.

    This function loads audio files in the list `audio_files` and concatenates
    audio data into one long array.
    If audio files in `audio_files` have sampling rate larger than 16 kHz,
    `ffmpeg` should be set to True to convert audio files to 16 kHz before loading.
    If the audio sampling rates are 16 kHz, `ffmpeg` can be set to False
    so that `librosa` can be used for audio loading.

    Args:
        audio_files: A list of paths to audio files, each file is 5-min long.
        ffmpeg: when True, use `ffmpeg` to load audio. Otherwise use `librosa`.
    """
    all_audio_data = []
    for file in audio_files:
        if ffmpeg:
            # Using ffmpeg to convert to 16kHz audio data and load the data
            audio_data, sr = utils.ffmpeg_load_audio(file)
        else:
            # Using librosa to directly load audios with native sampling rate
            audio_data, sr = librosa.load(file, sr=None)
        audio_data = round_data(audio_data)
        all_audio_data.append(audio_data)
    all_audio_data = np.array(all_audio_data).reshape(-1)

    return all_audio_data, sr


def preprocess(audio_file, args):
    # Load audio
    if isinstance(audio_file, list):
        audio_data, sr = load_audio_list(audio_file)
    elif args.ffmpeg:
        audio_data, sr = utils.ffmpeg_load_audio(audio_file)
    else:
        audio_data, sr = librosa.load(audio_file, sr=None)

    # Check for muted audio data
    if audio_data.sum() == 0:
        raise ValueError(f"Input audio is muted, audio path: {audio_file}")

    # Noise reduction
    if args.noise_reduce_off:
        print("[WARNING] Noise reduction is off.")
    else:
        audio_data = adaptive_noise_reduce(
            audio_data,
            sr,
            args.segment_len,
            estimate_noise_method=noise_minimum_energy,
            reduce_noise_method=spectral_gating,
            smoothing=args.smoothing,
            audio_len=args.audio_len,
        )

    if args.task_type == "sam2sam":
        mel_spec_data = sam2sam_preprocess(audio_data, sr, args)
    else:
        mel_spec_data = seq2seq_preprocess(audio_data, sr, args)

    return np.array(mel_spec_data)

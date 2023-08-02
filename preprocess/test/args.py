import argparse
import json


def build_data_args(parser):
    group = parser.add_argument_group("Data")
    group.add_argument("--sample_time", type=int, default=30, metavar="N", help="Time in one sample (in seconds)")
    group.add_argument(
        "--segment_len",
        type=float,
        default=30,
        metavar="N",
        help="Time of each segment (in seconds) in which the noise is estimated",
    )
    group.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        metavar="N",
        help="Smoothing coefficient to smooth the transition between different noise estimations",
    )
    group.add_argument(
        "--n_mels",
        type=int,
        default=20,
        metavar="N",
        help="Number of evenly spaced frequencies to separate into in Mel Spectrogram",
    )
    group.add_argument(
        "--window_size",
        type=float,
        default=50e-3,
        metavar="N",
        help="Window size which determines n_fft in librosa.feature.melspectrogram",
    )
    group.add_argument(
        "--stride",
        type=float,
        default=25e-3,
        metavar="N",
        help="Stride which determines hop_length in librosa.feature.melspectrogram",
    )

    group.add_argument(
        "--audio_len", type=float, default=300, metavar="N", help="Length of each audio (in second) from MVP frontend"
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Single person mel")
    parser.add_argument(
        "--task_type",
        type=str,
        default="sam2sam",
        choices=["sam2sam", "seq2seq"],
        help="Choosing task type, sample-to-sample or sequence-to-sequence",
    )
    parser.add_argument(
        "--audio_type", type=str, default="smartphone", choices=["SNUBH", "MVP", "smartphone"], help="Audio type"
    )
    parser.add_argument("--out_path", type=str, default="SINGLE_PERSON_MEL", help="Mel spectrogram save path")
    parser.add_argument("--nmel", type=int, default=1, help="Number of single epoch Mel to concatenate in the figure")
    parser.add_argument("--pid", type=str, default=None, help="Patient ID")

    build_data_args(parser)

    return parser


def print_args(args):
    params = vars(args)
    print("Parsed training parameters:")
    print(json.dumps(params, indent=4))


if __name__ == "__main__":
    parser = get_parser()
    parser.print_help()
    args = parser.parse_args()
    print_args(args)

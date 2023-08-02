import os
import torch

import subprocess
import librosa
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
from soundsleep.datasets.transforms import RandomPitch, SeqRandomPitch
from soundsleep.utils.inference import get_patient_idx, get_audio_idx_list, get_file_index
from soundsleep.preprocess.end2end_preprocess import preprocess
from args import get_parser, print_args


def zfill_MVP(file_list):
    new_file_list = []
    for filename in file_list:
        temp_idx = filename.split(".")[0]
        temp_file = str(temp_idx.zfill(4)) + ".mp3"
        new_file_list.append(temp_file)
    return new_file_list


def order_files(file_list):
    new_file_list = zfill_MVP(file_list)
    file_dict = dict(zip(new_file_list, file_list))
    new_file_list.sort()

    sorted_file_list = []
    for key in new_file_list:
        sorted_file_list.append(file_dict[key])

    return sorted_file_list


def draw(mel, label_list, save=False, path=None, show=False):
    plt.figure()

    # draw mel
    spec_db = librosa.power_to_db(mel, ref=np.max)
    specshow(spec_db, sr=16000, x_axis="time", y_axis="mel", hop_length=400, fmax=8000)
    plt.title("Mel Spectrogram {}".format(label_list))

    if save:
        plt.savefig(path)
    if show:
        plt.show()

    plt.close()


def draw_SNUBH(NMEL, DIR, PID, OUT_PATH, data_path):
    args.audio_len = None

    for i, dir_ in enumerate(os.listdir(data_path)):  # dir_ are data1, data2, and data3
        print(dir_)
        input_path = os.path.join(data_path, dir_, "test")

        patient_idx_list = get_patient_idx(input_path)
        print(patient_idx_list)

        if dir_ == DIR:
            file_list = [os.path.join(input_path, x) for x in os.listdir(input_path)]
            # Create loader for one patient
            file_list_idx = [x for x in file_list if "{:03d}_data".format(PID) in x]

            # Warning if the audio data is not contiguous
            audio_idx_list = get_audio_idx_list(file_list_idx)
            if (np.max(audio_idx_list) - np.min(audio_idx_list) + 1) != len(audio_idx_list):
                log = "[{}_data] Audio data is not contiguous. audio_idx_list is {}".format(idx, audio_idx_list)
                print(log)
                logging.warning(log)

            # Make sure files in file_list is in the chronological order
            file_list_sorted = []
            for audio_idx in audio_idx_list:
                files = [x for x in file_list_idx if "{}_data_{}".format(PID, audio_idx) in x]
                files = sorted(files, key=get_file_index)
                file_list_sorted += files

    save_dir = os.path.join(OUT_PATH, DIR + "-" + str(PID))
    os.makedirs(save_dir, exist_ok=True)
    for f in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, f))

    count = 0
    mel_list = []
    y_list = []
    for i, file_path in enumerate(file_list_sorted):
        count += 1
        data_dict = np.load(file_path, allow_pickle=True).item()
        mel = data_dict["x"]
        y = data_dict["4c"]
        mel_list.append(mel)
        y_list.append(y)

        if count == NMEL:
            concat_mel = np.concatenate(mel_list, axis=1)
            save_path = os.path.join(save_dir, str(i) + ".png")
            draw(concat_mel, y_list, save=True, path=save_path, show=False)

            mel_list = []
            y_list = []
            count = 0


def draw_MVP(NMEL, PID, OUT_PATH, data_path):
    data_path = os.path.join(data_path, PID)

    if PID == "data":
        data_path_list = os.listdir(data_path)
        for temp_user in data_path_list:
            temp_user_path = os.path.join(data_path, temp_user)
            print(temp_user_path)
            user_path_list = os.listdir(temp_user_path)
            for temp_data in user_path_list:
                temp_data_path = os.path.join(temp_user_path, temp_data)
                file_list = [x for x in os.listdir(temp_data_path) if "mp3" in x or "mp4" in x or "wav" in x]
                print(temp_user)
                print(temp_data)
                print(file_list)
                file_list = order_files(file_list)
                file_list = [os.path.join(temp_data_path, x) for x in file_list]
                # mel_spec_data = preprocess(file_list, args)
                try:
                    mel_spec_data = preprocess(file_list, args)
                except (RuntimeError, EOFError) as e:
                    print("Error happened: {}".format(e))
                    continue

                save_dir = os.path.join(OUT_PATH, temp_user + "_" + temp_data)
                os.makedirs(save_dir, exist_ok=True)
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))

                count = 0
                mel_list = []
                y_list = []
                for i, mel in enumerate(mel_spec_data):
                    count += 1
                    mel_list.append(mel)
                    y_list.append(-1)  # TODO

                    if count == NMEL:
                        concat_mel = np.concatenate(mel_list, axis=1)
                        save_path = os.path.join(save_dir, str(i) + ".png")
                        draw(concat_mel, y_list, save=True, path=save_path, show=False)

                        mel_list = []
                        y_list = []
                        count = 0

    else:
        file_list = [x for x in os.listdir(data_path) if "mp3" in x or "mp4" in x or "wav" in x]
        file_list = order_files(file_list)
        file_list = [os.path.join(data_path, x) for x in file_list]
        mel_spec_data = preprocess(file_list, args)

        save_dir = os.path.join(OUT_PATH, PID)
        os.makedirs(save_dir, exist_ok=True)
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))

        count = 0
        mel_list = []
        y_list = []
        for i, mel in enumerate(mel_spec_data):
            count += 1
            mel_list.append(mel)
            y_list.append(-1)  # TODO

            if count == NMEL:
                concat_mel = np.concatenate(mel_list, axis=1)
                save_path = os.path.join(save_dir, str(i) + ".png")
                draw(concat_mel, y_list, save=True, path=save_path, show=False)

                mel_list = []
                y_list = []
                count = 0


def draw_smartphone(NMEL, PID, OUT_PATH, data_path):
    args.audio_len = None

    data_path = os.path.join(data_path, PID)

    if PID == "experiment_data":
        user_path_list = os.listdir(data_path)
        for temp_data in user_path_list:
            temp_data_path = os.path.join(data_path, temp_data)
            file_list = [x for x in os.listdir(temp_data_path) if "mp3" in x or "mp4" in x or "wav" in x]
            if len(file_list) == 1:
                file_name = os.path.join(temp_data_path, file_list[0])

                # Load audio
                out_file = "16kHz_audio.wav"
                try:
                    # Convert to 16 kHz and load
                    subprocess.run(
                        ["ffmpeg", "-i", file_name, "-ar", "16000", out_file],
                        timeout=90,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    log = "[ffmpeg] Conversion error happened when converting {} to 16 kHz version: {}.".format(
                        audio_file, e
                    )
                    print(log)
                    logging.critical(log)
                    if os.path.exists(out_file):
                        os.remove(out_file)
                    continue
                audio, sr = librosa.load(out_file, sr=None)

                mel_spec_data = preprocess(out_file, args)
                os.remove(out_file)

                save_dir = os.path.join(OUT_PATH, PID + temp_data)
                os.makedirs(save_dir, exist_ok=True)
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))

                count = 0
                mel_list = []
                y_list = []
                for i, mel in enumerate(mel_spec_data):
                    count += 1
                    mel_list.append(mel)
                    y_list.append(-1)  # TODO

                    if count == NMEL:
                        concat_mel = np.concatenate(mel_list, axis=1)
                        save_path = os.path.join(save_dir, str(i) + ".png")
                        draw(concat_mel, y_list, save=True, path=save_path, show=False)

                        mel_list = []
                        y_list = []
                        count = 0

    else:
        file_list = [x for x in os.listdir(data_path) if "mp3" in x or "mp4" in x or "wav" in x or "3gp" in x]
        if len(file_list) == 1:
            file_name = os.path.join(data_path, file_list[0])
            mel_spec_data = preprocess(file_name, args)

            save_dir = os.path.join(OUT_PATH, PID)
            os.makedirs(save_dir, exist_ok=True)
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))

            count = 0
            mel_list = []
            y_list = []
            for i, mel in enumerate(mel_spec_data):
                count += 1
                mel_list.append(mel)
                y_list.append(-1)  # TODO

                if count == NMEL:
                    concat_mel = np.concatenate(mel_list, axis=1)
                    save_path = os.path.join(save_dir, str(i) + ".png")
                    draw(concat_mel, y_list, save=True, path=save_path, show=False)

                    mel_list = []
                    y_list = []
                    count = 0
        else:
            for temp_file in file_list:
                file_name = os.path.join(data_path, temp_file)
                mel_spec_data = preprocess(file_name, args)

                save_dir = os.path.join(OUT_PATH, PID + "_" + temp_file)
                os.makedirs(save_dir, exist_ok=True)
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))

                count = 0
                mel_list = []
                y_list = []
                for i, mel in enumerate(mel_spec_data):
                    count += 1
                    mel_list.append(mel)
                    y_list.append(-1)  # TODO

                    if count == NMEL:
                        concat_mel = np.concatenate(mel_list, axis=1)
                        save_path = os.path.join(save_dir, str(i) + ".png")
                        draw(concat_mel, y_list, save=True, path=save_path, show=False)

                        mel_list = []
                        y_list = []
                        count = 0


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_args(args)

    TYPE = args.audio_type
    OUT_PATH = args.out_path
    NMEL = args.nmel

    if TYPE == "SNUBH":
        ### Args for SNUBH audio
        DIR = "data2"
        if args.pid:
            PID = int(args.pid)
        else:
            PID = 1264
        data_root = os.environ["SOUNDSLEEP_DATA_PATH"]
        data_path = os.path.join(data_root, "sam2sam_19_20_old")

        draw_SNUBH(NMEL, DIR, PID, OUT_PATH, data_path)

    elif TYPE == "MVP":
        ### Args for Asleep MVP audio
        if args.pid:
            PID = args.pid
        else:
            PID = "data"
        data_path = "/mnt/SSD/.data/mvp_debug_data/"

        draw_MVP(NMEL, PID, OUT_PATH, data_path)

    elif TYPE == "smartphone":
        ### Args for smartphone audio
        if args.pid:
            PID = args.pid
        else:
            PID = "experiment_data"
        data_path = "/mnt/SSD/.data/TF_data"

        draw_smartphone(NMEL, PID, OUT_PATH, data_path)

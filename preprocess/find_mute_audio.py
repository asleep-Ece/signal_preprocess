"""
Load .wma audio with ffmpeg or load uncompressed .wav audio and count zero to check whether it's muted audio

Muted audio had ~ 78 % zero when load with uncompressed .wav,
~ 45 % zero when load with ffmpeg

"""

import librosa
import numpy as np
import os

INPUT_PATHS = ["/mnt/d/complete", "/mnt/e/data1/complete", "/mnt/e/data2/complete"]
OUTPUT_FILES = ["./data3.txt", "./data1.txt", "./data2.txt"]


def count_zero(sig):
    return np.count_nonzero(sig == 0)


def print_zero_statistics(sig, verbose=False):
    print("Count start")
    sig_len = sig.shape[0]
    zero_count = count_zero(sig)
    zero_portion = int(zero_count / sig.shape[0] * 100)
    if verbose:
        print("Signal length:{}, number of zero: {}, {}%".format(sig_len, zero_count, zero_portion))

    return sig_len, zero_count, zero_portion


if __name__ == "__main__":
    run = True
    for i, (INPUT_PATH, OUTPUT_FILE) in enumerate(zip(INPUT_PATHS, OUTPUT_FILES)):
        dir_list = os.listdir(INPUT_PATH)
        # print (dir_list)

        f = open(OUTPUT_FILE, "w")
        f.write("{} {} {} {}\n".format("dir", "sig_len", "zero_count", "zero_portion"))
        for temp_dir in dir_list:
            # Condition for finidng 2019~2020 data
            if OUTPUT_FILE == "./data1.txt":
                pid = int(temp_dir[:-5])
                if pid >= 814 and pid <= 1114:
                    run = True
                else:
                    run = False
            elif OUTPUT_FILE == "./data2.txt":
                pid = int(temp_dir[:-5])
                if pid >= 970 and pid <= 1505:
                    run = True
                else:
                    run = False
            else:
                run = True

            if run:
                if "data" in temp_dir:
                    wma_path = os.path.join(INPUT_PATH, temp_dir, "audio_0.wma")
                    print(wma_path)
                else:
                    continue

                sig, sr = librosa.load(wma_path, sr=16000)
                sig_len, zero_count, zero_portion = print_zero_statistics(sig, True)
                f.write("{} {} {} {}\n".format(temp_dir, sig_len, zero_count, zero_portion))
                f.flush()
        f.close()

import pickle
from psgsync import PSG_split
import argparse
import os
import scipy
parser = argparse.ArgumentParser(description="PSG data preprocess")

def add_arguments(parser):
    parser.add_argument('--sampling_rate', type=int, default=250, help='Downsampling frequency')

    return parser


with open("data1_train_clips.pkl", 'rb') as r:
    p_id = pickle.load(r)
print(p_id.keys())


edf_dir='/nas/SNUBH-PSG_signal_extract/train_data/data1-650_data/data1-650_data_signal/data1-650_data.edf'
offset_dir='/nas/SNUBH-PSG_signal_extract/train_data/data1-650_data/650_data_offset.csv'
label_dir = '/nas/SNUBH-PSG_signal_extract/train_data/data1-650_data/650_data_sleep_labels.csv'
process_train = PSG_split(parser, mode='train')
psg_epochs, _ = process_train.calculate_data_offset(edf_dir, offset_dir, label_dir)

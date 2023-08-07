import os
import numpy as np
import pandas as pd
import mne
from pyedflib import highlevel
from glob import glob


def add_arguments(parser):
    parser.add_argument('--sampling_rate', type=int, default=250, help='Downsampling frequency')

    return parser

class PSG_split():
    def __init__(self, parser):
        super(PSG_split, self).__init__()

        parser = add_arguments(parser)
        self.args = parser.parse_args()

        DATA_DIR = '/nas/SNUBH-PSG_signal_extract/'
        OUTPUT_DIR = '/nas/SNUBH-PSG_signal_extract/signal_extract'

    def get_edf_dir(self, patient_num, mode='train_data'):
        # Get directory of the PSG edf file
        sub_edf_path = os.path.join(DATA_DIR, mode, patient_num)
        edf_dir = os.path.join(sub_edf_path, patient_num+'_signal', patient_num+'.edf')
        
        return edf_dir

    def load_psg_channel(self, edf_dir, patient, channels):
        # Load psg data with selected channels
        signal_by_channel = []
        signals = []
        raw = mne.io.read_raw_edf(edf_dir)
        for channel in channels:
            raw_channel_signal = raw.get_data(channel)
            signal_by_channel.append(raw_channel_signal)
        signals = np.stack(signals, axis=0)

        return signals

    def calculate_data_offset(psg_dir):
        '''Cutoff the offset between PSG start time and label start time'''
        pass

    def divide_psg_data():
        '''
        divide psg data into 30s with considering the frequency
            for data_dir in os.listdir(DATA_DIR):
            sub_edf_path = os.path.join(DATA_DIR, mode, data_dir)
            if not os.path.isdir(sub_edf_path):
                continue
            patient_list = [os.path.join(sub_edf_path, x) for x in os.listdir(sub_edf_path)]
            patient_list = [x for x in patient_list if os.path.isdir(x)]
            if len(patient_list) == 0:
                continue
        '''
        pass

    def check_disconnection():
        '''check whether there are disconnections by file name'''
        pass

    def check_xml():
        '''check start time of the disconnected xml file'''
        pass

    def calculate_label_starttime():
        '''Find the nearest 30x time from the start time of the xml file'''
        pass

    def test():
        pass
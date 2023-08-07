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
        epoch = 30
        '''divide psg data into 30s with considering the frequency'''
        f = pyedflib.EdfReader(psg_dir)
        for chn in range(f.signals_in_file):
            if f.getLabel(chn) in self.args.chns:
                #cal each chn freq
                raw_rate = f.getSampleFrequency(chn)
                #read data
                raw_data = f.readSignal(chn)
            print("Sfreq : {} | shape: {}".format(raw_rate,len(raw)))

            
            # clip start_dime offset
            # get the offset info
            label_start = pd.read_csv(offset_dir)["label_start"].values[0]
            raw_start = f.getStartdatetime()
            raw_start = datetime.datetime.strftime(raw_start,"%H:%M:%S")
            print("label start time: {} | edf start time: {}".format(label_start,raw_start))
            startime = ((datetime.datetime.strptime(label_start,"%H:%M:%S")-datetime.datetime.strptime(raw_start,"%H:%M:%S")).seconds)*int(raw_rate)
            raw_data = raw_data[startime:]

            #get the labels
            labels = get_labels(psg_dir)

            #check if the psg data length > expected lenght (num of labels x 30 seconds)
            flag = len(raw_data)- labels*epoch*raw_rate

            if flag == 0:
                pass
            elif flag > 0:
                raw_data = raw_data[:-flag]
            else:
                # Discard redundant labels and corresponding data
                red_labels = (-flag/epoch).ceil()
                raw_data = raw_data[:-red_labels*epoch]
                labels = labels[:-red_labels]

            # divide into 30 seconds based on the number of labels
            raw_data_epochs = np.array_split(raw_data, len(labels))


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
import os
import numpy as np
import pandas as pd
import mne
from pyedflib import highlevel
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="PSG data preprocess")

def add_arguments(parser):
    parser.add_argument('--sampling_rate', type=int, default=250, help='Downsampling frequency')

    return parser

class PSG_split():
    def __init__(self, parser):
        super(PSG_split, self).__init__()

        parser = add_arguments(parser)
        self.args = parser.parse_args()
        sam_rate = self.args.sampling_rate

        self.DATA_DIR = '/nas/SNUBH-PSG_signal_extract/'
        self.OUTPUT_DIR = '/nas/SNUBH-PSG_signal_extract/signal_extract'

    def get_edf_dir(self, patient_num, mode='train_data'):
        # Get directory of the PSG edf file
        sub_edf_path = os.path.join(self.DATA_DIR, mode, patient_num)
        edf_dir = os.path.join(sub_edf_path, patient_num+'_signal', patient_num+'.edf')
        # Check if there is edf file in the directory
        if not os.path.isfile(edf_dir):
            print(f'Patient {patient_num} has no edf file. Skipping...')
        # If True, return offset, edf, label directory
        else:
            offset_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_sleep_labels.csv')
            return edf_dir, offset_dir, label_dir

    def calculate_data_offset(edf_dir,offset_dir,label_dir):
        '''
        1. Cutoff the offset between PSG start time and label start time
        2. Remove the end rebundent labels and data
        3. spilit data into 30 seconds
        
        return:
            psg_epochs: processed chns data (length should be chns)
            psg_names : the names of chns(length should be chns)
            labels : the processed labels
        
        '''
        epoch = 30
        psg_epochs = []
        psg_names = []
        '''divide psg data into 30s with considering the frequency'''
        f = pyedflib.EdfReader(edf_dir)
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
            labels = pd.read_csv(label_dir,header=None).values

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

            psg_epochs.add(raw_data_epochs)
            psg_names.add(f.getLabel(chn))

        #return the processed data(chns) from the current patient
        return psg_epochs,psg_names,labels



    def save_psg_data(self, psg_epochs,psg_names,labels):
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
        ''' 
            Save each patient's data every 30seconds
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
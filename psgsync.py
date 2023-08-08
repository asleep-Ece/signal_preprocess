import os
import numpy as np
import pandas as pd
import mne
from pyedflib import highlevel
from glob import glob
import argparse
import pyedflib
import datetime
import math
import pickle

parser = argparse.ArgumentParser(description="PSG data preprocess")

def add_arguments(parser):
    parser.add_argument('--sampling_rate', type=int, default=250, help='Downsampling frequency')

    return parser

class PSG_split():
    def __init__(self, parser):
        super(PSG_split, self).__init__()

        parser = add_arguments(parser)
        self.args = parser.parse_args()

        self.DATA_DIR = '/nas/SNUBH-PSG_signal_extract/'
        self.OUTPUT_DIR = '/nas/SNUBH-PSG_signal_extract/signal_extract/'
        self.chns = ['Plethysmogram', 'A1']

    def get_edf_dir(self, patient_num, mode='train_data'):
        # Get directory of the PSG edf file
        sub_edf_path = os.path.join(self.DATA_DIR, mode, patient_num)
        edf_dir = os.path.join(sub_edf_path, patient_num+'_signal', patient_num+'.edf')
        # Check if there is edf file in the directory
        if not os.path.isfile(edf_dir):
            print(f'Patient {patient_num} has no edf file. Skipping...')
        # If True, return offset, edf, label directory
        else:
            if len(patient_num.split('-')[1].split('_')[0])==1:
                offset_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_offset.csv')
                label_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_sleep_labels.csv')
            elif len(patient_num.split('-')[1].split('_')[0])==2:
                offset_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_offset.csv')
                label_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_sleep_labels.csv')
            elif len(patient_num.split('-')[1].split('_')[0])<=3:
                offset_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_offset.csv')
                label_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_sleep_labels.csv')
            return edf_dir, offset_dir, label_dir

    def calculate_data_offset(self, edf_dir,offset_dir,label_dir):
        '''
        1. Cutoff the offset between PSG start time and label start time
        2. Remove the end redundent labels and data
        3. split data into 30 seconds
        
        return:
            psg_epochs: processed chns data (len(psg_epochs) should be #chns)
            psg_names : the names of chns(len(psg_names) should be #chns)
            labels : the processed labels
        
        '''
        epoch = 30
        psg_epochs = dict()
        '''divide psg data into 30s with considering the frequency'''
        f = pyedflib.EdfReader(edf_dir)
        for chn in range(f.signals_in_file):
            if f.getLabel(chn) in self.chns:
                #cal each chn freq
                raw_rate = f.getSampleFrequency(chn)
                #read data
                raw_data = f.readSignal(chn)
                print("Sfreq : {} | shape: {}".format(raw_rate,len(raw_data)))

                
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
                flag = len(raw_data)- len(labels)*epoch*raw_rate

                if flag == 0:
                    pass
                elif flag > 0:
                    raw_data = raw_data[:-flag]
                else:
                    # Discard redundant labels and corresponding data
                    red_labels = math.ceil(-flag/epoch)
                    raw_data = raw_data[:-red_labels*epoch]
                    

                # divide into 30 seconds based on the number of labels
                raw_data_epochs = np.array_split(raw_data, len(labels))

                # psg_epochs.append(raw_data_epochs)
                psg_epochs[f.getLabel(chn)] = raw_data_epochs
            # psg_names.append(f.getLabel(chn))
        labels = labels[:-red_labels]
        #return the processed data(chns) from the current patient
        return psg_epochs,labels

    def save_one_psg(self, patient_num, psg_epochs, labels, mode='train'):
        # patient_num : data1-73_data
        data_group = patient_num.split('-')[0]
        os.makedirs(os.path.join(self.OUTPUT_DIR,data_group,mode), exist_ok=True)
    
        split_psg_dir = os.path.join(self.OUTPUT_DIR,data_group,mode,patient_num.split('-')[1]+'_0_')
        for idx in range(len(list(psg_epochs.values())[0])):
            split_psg = {key:list(value[idx]) for key, value in psg_epochs.items()}
            with open(split_psg_dir+str(idx)+'.pickle', 'wb') as fw:
                pickle.dump(split_psg, fw)

        # for idx in range(len(list(psg_epochs.values())[0])):
        #     for ch in psg_epochs.keys():
        #         with open(split_psg_dir+str(idx),'ab') as fw:
        #             pickle.dump(psg_epochs.values()[idx], fw)

        # for i, data in enumerate(list(psg_epochs.values())[0]):
        #     for ch in psg_epochs.keys():
        #         with open(split_psg_dir+str(i),'wb') as fw:
        #             pickle.dump(data, fw)
            # np.save(split_psg_dir+str(i), data)



    def save_psg_data(self, psg_epochs,psg_names,labels):
        '''
        divide psg data into 30s with considering the frequencㅛ 
        Save each patient's data every 30seconds
        '''
        for data_dir in os.listdir(self.DATA_DIR):
            sub_edf_path = os.path.join(self.DATA_DIR, mode, data_dir)
            if not os.path.isdir(sub_edf_path):
                continue
            patient_list = [os.path.join(sub_edf_path, x) for x in os.listdir(sub_edf_path)]
            patient_list = [x for x in patient_list if os.path.isdir(x)]
            if len(patient_list) == 0:
                continue

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

a = PSG_split(parser)
x,y,z = a.get_edf_dir('data1-73_data')
print(x,y,z)
psg_epoch, label = a.calculate_data_offset(x,y,z)
print(len(list(psg_epoch.values())[0]))
a.save_one_psg('data1-73_data', psg_epoch, label)
with open('/nas/SNUBH-PSG_signal_extract/signal_extract/data1/train/73_data_0_0', 'rb') as fr:
    a = pickle.load(fr)
    print('length : ', len(a['Plethysmogram']), len(a['A1']), a)
# print(len(k.item().get('Plethysmogram')))
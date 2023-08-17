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
import re
import time

parser = argparse.ArgumentParser(description="PSG data preprocess")

def add_arguments(parser):
    parser.add_argument('--sampling_rate', type=int, default=250, help='Downsampling frequency')

    return parser

class PSG_split():
    def __init__(self, parser, mode):
        super(PSG_split, self).__init__()

        parser = add_arguments(parser)
        self.args = parser.parse_args()
        self.mode = mode    # 'train' or 'test'

        self.DATA_DIR = '/nas/SNUBH-PSG_signal_extract/'
        self.OUTPUT_DIR = '/nas/SNUBH-PSG_signal_extract/signal_extract/'
        self.SOUND_DIR = '/nas/max/tmp_data/dataset_abcd/psg_abc'
        self.chns = ['Plethysmogram', 'C3-A2','C4-A1','F3-A2','F4-A1','LOC','ROC','O1-A2','O2-A1']

    def get_edf_dir(self, sub_edf_path, patient_num):
        p = patient_num.split('-')[1].split('_')[0]
        # Some offset and labels csv files have 0 in front of the file name
        if len(p)==1:
            offset_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_sleep_labels.csv')
        elif len(p)==2:
            offset_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_sleep_labels.csv')
        elif len(p)>=3:
            offset_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, patient_num.split('-')[1]+'_sleep_labels.csv')
        return offset_dir, label_dir

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
        #get the labels
        labels = pd.read_csv(label_dir,header=None).values

        '''divide psg data into 30s with considering the frequency'''
        f = pyedflib.EdfReader(edf_dir)
        for chn in range(f.signals_in_file):
            temp_labels = labels
            if f.getLabel(chn) in self.chns:
                #cal each chn freq
                raw_rate = f.getSampleFrequency(chn)
                #read data
                raw_data = f.readSignal(chn)
                # print("Sfreq : {} | shape: {}".format(raw_rate,len(raw_data)))

                
                # clip start_dime offset
                # get the offset info
                label_start = pd.read_csv(offset_dir)["label_start"].values[0]
                raw_start = f.getStartdatetime()
                raw_start = datetime.datetime.strftime(raw_start,"%H:%M:%S")
                # print("label start time: {} | edf start time: {}".format(label_start,raw_start))
                startime = ((datetime.datetime.strptime(label_start,"%H:%M:%S")-datetime.datetime.strptime(raw_start,"%H:%M:%S")).seconds)*int(raw_rate)
                raw_data = raw_data[startime:]
                # print(f"startoff data lenth {len(raw_data)}")

                
                #check if the psg data length > expected lenght (num of labels x 30 seconds)
                flag = len(raw_data)- len(labels)*epoch*raw_rate
                

                if flag == 0:
                    pass
                elif flag > 0:
                    raw_data = raw_data[:-int(flag)]
                else:
                    # Discard redundant labels and corresponding data
                    red_labels = math.ceil(-flag/(epoch*raw_rate))
                    temp_labels = temp_labels[:-red_labels]
                    print(f"offset: {-flag}, red_labels {red_labels} rate {raw_rate}")
                    edd_off = len(raw_data)-len(temp_labels)*epoch*int(raw_rate)
                    raw_data = raw_data[:-edd_off]
                    print(f"processed data: {len(raw_data)}")
                    
                # divide into 30 seconds based on the number of labels
                raw_data_epochs = np.split(raw_data, len(temp_labels))
                # print(f"1st data {len(raw_data_epochs[0])} last data {len(raw_data_epochs[-1])}")
                # psg_epochs.append(raw_data_epochs)
                psg_epochs[f.getLabel(chn)] = raw_data_epochs

            # psg_names.append(f.getLabel(chn))
        
        #return the processed data(chns) from the current patient
        return psg_epochs,temp_labels

    def save_one_psg(self, patient_num, psg_epochs):
        # patient_num format : data1-73_data
        data_group = patient_num.split('-')[0]
        os.makedirs(os.path.join(self.OUTPUT_DIR,data_group,self.mode), exist_ok=True)
    
        split_psg_dir = os.path.join(self.OUTPUT_DIR,data_group,self.mode,patient_num.split('-')[1]+'_0_')

        print(f"=============")
        print(f"total idx : {len(list(psg_epochs.values())[0])}")

        for idx in range(len(list(psg_epochs.values())[0])):
            split_psg = {key:list(value[idx]) for key, value in psg_epochs.items()} 

            with open(split_psg_dir+str(idx)+'.pkl', 'wb') as fw:
                pickle.dump(split_psg, fw)


    def save_all_psg(self):
        '''
        divide psg data into 30s with considering the frequency
        Save each patient's data every 30seconds
        '''
        # Get directory of the PSG edf file
        for patient_num in os.listdir(os.path.join(self.DATA_DIR, self.mode+'_data')):
            sub_edf_path = os.path.join(self.DATA_DIR, self.mode+'_data', patient_num)
            if not os.path.isdir(sub_edf_path):
                continue
            edf_dir = os.path.join(sub_edf_path, patient_num+'_signal', patient_num+'.edf')
            # Check if there is edf file in the directory
            if not os.path.isfile(edf_dir):
                print(f'Patient {patient_num} has no edf file. Skipping...')
                continue
            else:
                offset_dir, label_dir = self.get_edf_dir(sub_edf_path, patient_num)
                psg_epochs, _ = self.calculate_data_offset(edf_dir, offset_dir, label_dir)
                self.save_one_psg(patient_num, psg_epochs)
            print(f'Patient {patient_num} has been successfully saved')

    def check_disconnection(self, group):
        '''
        Check whether there are disconnections by file name
        Find out all disconnections with patients_id
        Return the duration of each disconnected sound data for each patient
        
        return:
            clips (dictionary) : 
                key : patient id
                values : duration of each disconnected audio
        '''
        # Get all patient num from the sound data and check if there's identical one in PSG
        sound_patient_list = []
        psg_patient_list = []
        disconnection_count = dict()
        clips = dict()
        pattern = r'\d+'
        group_sound_path = os.path.join(self.SOUND_DIR, group, self.mode)
        # Save patient_list 
        if group in os.listdir(self.OUTPUT_DIR):
            for i in os.listdir(group_sound_path):
                sound_patient_list.append(i.split('_')[0])
            for i in os.listdir(os.path.join(self.OUTPUT_DIR, group, self.mode)):
                psg_patient_list.append(i.split('_')[0])
        sound_patient_list = list(set(sound_patient_list))
        psg_patient_list = list(set(psg_patient_list))
        # Check disconnected 
        for i in psg_patient_list:
            # Check if there's identical patient in both data
            if i in sound_patient_list:
                duration = []
                clip_num = dict()
                for j in os.listdir(group_sound_path):
                    # Get number of each disconnected data for each patient
                    if i==re.findall(pattern,j)[0] :
                        if re.findall(pattern,j)[1] not in clip_num.keys() or clip_num[re.findall(pattern,j)[1]]<=int(re.findall(pattern,j)[2]):
                            clip_num[re.findall(pattern,j)[1]] = int(re.findall(pattern,j)[2])
                # Only get disconnected patient
                if len(clip_num.keys())>1:
                    k = sorted(list(clip_num.keys()))
                    for key in k:
                        duration.append(clip_num[key]*30)
                    clips[int(i)] = duration
                else:
                    print(f"{i} patient haven't disconnected")
                    continue
            else:
                print(f'No {i} patient in {group}')
                continue
            
        return clips

    def check_xml(self, group ,p_id):
        '''
            extract start time of the disconnected xml file

        return:
            clip_times: global start ime after each disconecting moments

        '''
        clip_times = [] 

        p_dir = os.path.join(self.DATA_DIR,f"{self.mode}_data", f"{group}-{p_id}_data")
        pattern = r"video_(\d+).xml"

        for f in os.listdir(p_dir):
            match = re.match(pattern, f)
            if match:
                # print(f)
                #extract begin time
                xml_dir = os.path.join(p_dir,f)
                # print(xml_dir)
                # read XML
                with open(xml_dir, 'r') as r:
                    xml_content = r.read()

                # matching time
                p = r"<Start>(.*?)<\/Start>"
                matches = re.findall(p, xml_content)
                # print(matches)

                if matches:
                    for match in matches:
                    #  %H:%M:%S output
                        time_format = re.search(r"\d{2}:\d{2}:\d{2}", match)
                        if time_format:
                            extracted_time = time_format.group()
                            clip_times.append(extracted_time)

                else:
                    print(xml_dir)
                    print("No <Start> elements with time found in the XML.")
            else:
                pass

        return clip_times

    def check_label_start(self, group, p_id):
        offset_dir = os.path.join(self.DATA_DIR, f"{self.mode}_data", f"{group}-{p_id}_data",f"{p_id}_data_offset.csv")
        label_start = pd.read_csv(offset_dir)["label_start"].values[0]
        return label_start

    def calculate_disconnection(self,group):

        epoch = 30
        # p_id,num_audio,clips,c_times
        dur_time = {}
        '''
        Find the nearest 30x time from the start time of the xml file
        
        return:
             duration_starts:
             durations: the disconnecting duration in data(notice: may disconnected few times)
             num_disconnections: how many files are dismissed
        
        '''      

        #open corresponding pkl info
        pkl_dir = os.path.join(f"{group}_{self.mode}_clips.pkl")
        with open(pkl_dir, 'rb') as f:
            clip = pickle.load(f)

        #calculate each patient's disconnection's num_epoch
        for k in clip.keys():
            #label start time as a standaration
            start = self.check_label_start(group,k)
            # get xml time 
            clip_times = self.check_xml(group,k)
            dur_times=[]
            clip_times[0]=start
            for i in range(len(clip[k])-1):
                clip[k][i] = clip[k][i]+epoch
                clipEnd = datetime.datetime.strptime(clip_times[i],"%H:%M:%S")+datetime.timedelta(seconds = (clip[k][i]))
                clipEnd = datetime.datetime.strftime(clipEnd,"%H:%M:%S")
                # print(f"clipEnd {clipEnd}")

                # print(f"clip_times {clip_times[i+1]}")
                duration = (datetime.datetime.strptime(clip_times[i+1],"%H:%M:%S")-datetime.datetime.strptime(clipEnd,"%H:%M:%S")).seconds
                # print(duration)

                num_disconnections = math.ceil(duration/epoch)

                dur_times.append(num_disconnections)
                clip[k][i]= int(clip[k][i] / epoch)
                clip[k][i] += num_disconnections
            clip[k][-1]=int(clip[k][-1] / epoch)
            dur_time[k]=dur_times
        return dur_time,clip

    def rename_file(self,group):
        psg_dir = os.path.join(self.OUTPUT_DIR,group,self.mode)
        _,clipped = self.calculate_disconnection(group)

        # Get all split sessions for each patient
        for p_id in clipped.keys():
            print(clipped[p_id])
            print(p_id)
            p_dirs=[]
            psg_pattern = f"{p_id}_data_(\d+)_(\d+).pkl"
            psg_patients = []
            patient_data = []
            for f in os.listdir(psg_dir):
                match = re.match(psg_pattern,f)
                if match:
                    patient_data.append(f)
                else:
                    pass

            # print(patient_data)
            print("================================")
            # Rename the pkl files whenever there are disconnections
            for p in range(len(patient_data)):
                global_count=0
                for i, data in enumerate(clipped[p_id]):
                    for j in range(data):
                        f_name=f"{p_id}_data_0_{global_count}.pkl"
                        if f_name in patient_data[p]:
                            # patient_data[p]=f"{p_id}_data_{i}_{j}.pickle"
                            os.rename(os.path.join(psg_dir,f_name),os.path.join(psg_dir,f"{p_id}_data_{i}_{j}.pkl"))
                        global_count+=1  
            # print(patient_data)                             
            # break


if __name__ == "__main__":
    SOUND_DIR = '/nas/max/tmp_data/dataset_abcd/psg_abc'
    process_train = PSG_split(parser, mode='train')
    parser = argparse.ArgumentParser(description="PSG data preprocess")
    process_test = PSG_split(parser, mode='test')

    # Save all psg data to OUTPUT_DIR
    process_train.save_all_psg()
    process_test.save_all_psg()

    #Save clips in pkl format
    for group in os.listdir(SOUND_DIR):  # Currently there are upto data5 group
        clips_train = process_train.check_disconnection(group)
        clips_test = process_test.check_disconnection(group)
        
        with open(f'{group}_train_clips.pkl', 'wb') as fw:
            pickle.dump(clips_train, fw)
        with open(f'{group}_test_clips.pkl', 'wb') as fw:
            pickle.dump(clips_test, fw)
        
        clips_train.rename_file(group)
        clips_test.rename_file(group)


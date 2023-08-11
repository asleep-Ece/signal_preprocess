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
    def __init__(self, parser):
        super(PSG_split, self).__init__()

        parser = add_arguments(parser)
        self.args = parser.parse_args()

        self.DATA_DIR = '/nas/SNUBH-PSG_signal_extract/'
        self.OUTPUT_DIR = '/nas/SNUBH-PSG_signal_extract/signal_extract/'
        self.SOUND_DIR = '/nas/max/tmp_data/dataset_abcd/psg_abc'
        self.chns = ['Plethysmogram', 'A1']

    def get_edf_dir(self, sub_edf_path, patient_num):
        if len(patient_num.split('-')[1].split('_')[0])==1:
            offset_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, '00'+patient_num.split('-')[1]+'_sleep_labels.csv')
        elif len(patient_num.split('-')[1].split('_')[0])==2:
            offset_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_offset.csv')
            label_dir = os.path.join(sub_edf_path, '0'+patient_num.split('-')[1]+'_sleep_labels.csv')
        elif len(patient_num.split('-')[1].split('_')[0])>=3:
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
                print("Sfreq : {} | shape: {}".format(raw_rate,len(raw_data)))

                
                # clip start_dime offset
                # get the offset info
                label_start = pd.read_csv(offset_dir)["label_start"].values[0]
                raw_start = f.getStartdatetime()
                raw_start = datetime.datetime.strftime(raw_start,"%H:%M:%S")
                print("label start time: {} | edf start time: {}".format(label_start,raw_start))
                startime = ((datetime.datetime.strptime(label_start,"%H:%M:%S")-datetime.datetime.strptime(raw_start,"%H:%M:%S")).seconds)*int(raw_rate)
                raw_data = raw_data[startime:]
                print(f"startoff data lenth {len(raw_data)}")

                
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
                print(f"1st data {len(raw_data_epochs[0])} last data {len(raw_data_epochs[-1])}")
                # psg_epochs.append(raw_data_epochs)
                psg_epochs[f.getLabel(chn)] = raw_data_epochs

            # psg_names.append(f.getLabel(chn))
        
        #return the processed data(chns) from the current patient
        return psg_epochs,temp_labels

    def save_one_psg(self, patient_num, psg_epochs, mode):
        # patient_num : data1-73_data
        data_group = patient_num.split('-')[0]
        os.makedirs(os.path.join(self.OUTPUT_DIR,data_group,mode), exist_ok=True)
    
        split_psg_dir = os.path.join(self.OUTPUT_DIR,data_group,mode,patient_num.split('-')[1]+'_0_')

        print(f"=============")
        print(f"total idx : {len(list(psg_epochs.values())[0])}")

        for idx in range(len(list(psg_epochs.values())[0])):
            split_psg = {key:list(value[idx]) for key, value in psg_epochs.items()} 

            with open(split_psg_dir+str(idx)+'.pkl', 'wb') as fw:
                pickle.dump(split_psg, fw)


    def save_all_psg(self, mode='train'):
        '''
        divide psg data into 30s with considering the frequency
        Save each patient's data every 30seconds
        '''
        # Get directory of the PSG edf file
        for patient_num in os.listdir(os.path.join(self.DATA_DIR, mode+'_data')):
            sub_edf_path = os.path.join(self.DATA_DIR, mode+'_data', patient_num)
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
                self.save_one_psg(patient_num, psg_epochs, mode=mode)
            print(f'Patient {patient_num} has been successfully saved')

    def check_disconnection(self, group, mode='train'):
        '''
        check whether there are disconnections by file name
        find out all disconnections patients_id
        /nas/max/temp-data/~~ 에서 disconnected patient list 찾아서 PSG data에 같이 포함되는 데이터만 찾기
        
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
        group_sound_path = os.path.join(self.SOUND_DIR, group, mode)
        # Save patient_list 
        if group in os.listdir(self.OUTPUT_DIR):
            for i in os.listdir(group_sound_path):
                sound_patient_list.append(i.split('_')[0])
            for i in os.listdir(os.path.join(self.OUTPUT_DIR, group, mode)):
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
                    if i==re.findall(r'\d+',j)[0] : # 모든 patient_id가 같은애들에 대해서  and re.findall(r'\d+',j)[1]!=0
                        if re.findall(r'\d+',j)[1] not in clip_num.keys() or clip_num[re.findall(r'\d+',j)[1]]<=int(re.findall(r'\d+',j)[2]):
                            clip_num[re.findall(r'\d+',j)[1]] = int(re.findall(r'\d+',j)[2])
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

    def check_xml(self, group_id,p_id,mode='train'):
        '''
            extract start time of the disconnected xml file

        return:
            clip_times: global start ime after each disconecting moments

        '''
        clip_times = [] 

        p_dir = os.path.join(self.DATA_DIR,f"{mode}_data", f"data{group_id}-{p_id}_data")
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

    def check_label_start(self, group_id, p_id,mode='train'):
        offset_dir = os.path.join(self.DATA_DIR, f"{mode}_data", f"data{group_id}-{p_id}_data",f"{p_id}_data_offset.csv")
        label_start = pd.read_csv(offset_dir)["label_start"].values[0]
        return label_start

    def calculate_disconnection(self,group_id):

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
        pkl_dir = os.path.join(f"data{group_id}_train_clips.pkl")
        with open(pkl_dir, 'rb') as f:
            a = pickle.load(f)

        #calculate each patient's disconnection's num_epoch
        for k in a.keys():
            #label start time as a standaration
            start = self.check_label_start(group_id,k)
            # get xml time 
            clip_times = self.check_xml(group_id,k,mode='train')
            dur_times=[]
            clip_times[0]=start
            for i in range(len(a[k])-1):
                a[k][i] = a[k][i]+epoch
                clipEnd = datetime.datetime.strptime(clip_times[i],"%H:%M:%S")+datetime.timedelta(seconds = (a[k][i]))
                clipEnd = datetime.datetime.strftime(clipEnd,"%H:%M:%S")
                # print(f"clipEnd {clipEnd}")

                # print(f"clip_times {clip_times[i+1]}")
                duration = (datetime.datetime.strptime(clip_times[i+1],"%H:%M:%S")-datetime.datetime.strptime(clipEnd,"%H:%M:%S")).seconds
                # print(duration)

                num_disconnections = math.ceil(duration/epoch)

                dur_times.append(num_disconnections)
                a[k][i]= int(a[k][i] / epoch)
                a[k][i] += num_disconnections
            a[k][-1]=int(a[k][-1] / epoch)
            dur_time[k]=dur_times
            print("dis number : ",dur_time[k])
            print("duration :" , a[k])
            # break
        
        return dur_time,a
    


a = PSG_split(parser)
# a.save_all_psg(mode='train')
# Save clips in pkl format
# for i in range(1,4):
#     clips = a.check_disconnection('data'+str(i))
#     with open('data'+str(i)+'_train_clips.pkl', 'wb') as fw:
#         pickle.dump(clips, fw)
    
# with open('/nas/SNUBH-PSG_signal_extract/signal_extract/data1/train/73_data_0_1012.pickle', 'rb') as fr:
#     a = pickle.load(fr)
#     print('length : ', len(a['Plethysmogram']), len(a['A1']), len(a))

# a.check_disconnection_1()
# a.check_disconnection("data1", mode='train')
# print(a.check_xml(1,449))

dur_time = a.calculate_disconnection(1)

# for i in dur_time.items():
#     print(i)
#     break


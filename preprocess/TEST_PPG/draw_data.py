import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyedflib
import time
import tsfel
# current path
save_path=os.getcwd()
print(save_path)



def get_test_ppg_data(patient_dir):
    #get ppg from raw data 
    raw = mne.io.read_raw_edf(f"{patient_dir}/data3-436_data/data3-436_data_signal/data3-436_data.edf")
    channels=raw.ch_names
    f = pyedflib.EdfReader(f"{patient_dir}/data3-436_data/data3-436_data_signal/data3-436_data.edf")
    labels=pd.read_csv(f"{patient_dir}/data3-436_data/436_data_sleep_labels.csv")
    print("label shape: ",labels.shape)
    for i, ch_name in enumerate(channels):

        # raw_signal = raw.get_data(ch_name)
        # print(f"{ch_name} data shape:{raw_signal.shape}")
        if ch_name=='Plethysmogram':
            sample_rate=f.getSampleFrequency(i)
            raw_signal = f.readSignal(i)
            print(f"{ch_name} sample_rate frequency:{sample_rate}")
            print(f"{ch_name} data shape:{raw_signal.shape}")
        
    
    # raw_signal=raw_signal.transpose()
    
    return True

def read_edf_file(patient_dir):
    raw_signal_list=[]
    signal_wave=os.path.join(parent_dir,"data3-9_data_signal/data3-9_data.edf")

    f = pyedflib.EdfReader(signal_wave)
    n = f.signals_in_file
    print("signal numbers:", n)
    signal_labels = f.getSignalLabels()
    print("Labels:", signal_labels, len(signal_labels), sep="\n==========\n")
    sample_signal=f.getSampleFrequencies()
    print("startdate: %i-%i-%i" %
          (f.getStartdatetime().day, f.getStartdatetime().month,
           f.getStartdatetime().year))
    print("starttime: %i:%02i:%02i" %
    (f.getStartdatetime().hour, f.getStartdatetime().minute,
    f.getStartdatetime().second))

    # for i in np.arange(n):
    #     sigbufs[i, :] = f.readSignal(i)
    #     print('ooo',sigbufs)
    #     label = f.getLabel(i)
    #     sample_rate=f.getSampleFrequency(i)
    #     # print('sample_rate',sample_rate)
    #     d = list(enumerate(sigbufs[0, 0:sample_rate]))
    #     print('da',d)
        # print({label: sigbufs[0, 0:10]})
        # return {label: sigbufs[0, 0:256]}
    # print(sigbufs.shape)

    # return sigbufs

#extract feature of PPG by using TSFEL
def extract_feature_ppg(patient_dir):
    raw_signal = get_test_ppg_data(patient_dir)
    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    # Extract features
    X = tsfel.time_series_features_extractor(cfg, raw_signal)
    return X



# get_labels
def get_test_ppg_label(patient_dir):
    #validate the number of labels 
    labels=pd.read_csv(open(f"{patient_dir}/sleep_labels.csv"))
    print(labels.shape)
    
    return True

#plot
def plot_ppg(save_path,raw_signal):
    #show the PPG data
    
    return True

#caculate label
parent_dir='/nas/SNUBH-PSG_signal_extract/train_data/'

# ppg_feature= extract_feature_ppg(parent_dir)
# print(ppg_feature.shape)
get_test_ppg_data(parent_dir)
# plot_ppg(parent_dir,save_path)
# get_test_ppg_label(parent_dir)
# calculate_data_offset(parent_dir,save_path)
# read_edf_file(parent_dir)
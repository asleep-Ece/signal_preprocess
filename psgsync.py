import os
import numpy as np
import pandas as pd

'''우리가 하려고 하는것
1. psg signal의 start time과 sound의 label start time을 일단 맞춘다
2. psg signal을 가지고 원하는 channel에 대한 데이터를 추출하고 
3. 30초씩 잘라서 numpy로 저장 (nas/SNUBH-PSG_~/signal_extract)에다가 기존의 형식대로 그대로 디렉토리 만들어서 저장
4. 30초씩 잘려진 sound data의 파일명이 0에서 1로 바뀐다면 그 사이의 애들은 버리고 그 뒤의 애들은 전부 rename
''' 

def load_psg_channel():
    '''Load psg file with selected channels'''
    pass

def calculate_data_offset(psg_dir):
    '''Cutoff the offset between PSG start time and label start time'''
    pass

def divide_psg_data():
    '''divide psg data into 30s with considering the frequency'''
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


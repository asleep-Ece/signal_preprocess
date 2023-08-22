import os 

data_dir = '/nas/SNUBH-PSG_signal_extract/signal_extract/data1/test'

count = 0
for i in os.listdir(data_dir):
    count+=1

print(count)

# data_dir = '/nas/max/tmp_data/dataset_abcd/psg_abc/data1/train'

# count=0
# for i in os.listdir(data_dir):
#     count+=1

# print(count)
# num = 0
# for i in os.listdir(data_dir):
#     p_id = int(i.split('_')[0])
#     if num<=p_id:
#         num=p_id

# print(num)
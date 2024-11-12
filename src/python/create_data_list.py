import os
import numpy as np

data_folder_path = '/home/user/work/data/Data_20240705/3-51200/linear/raws'

f = open('dataset_iso51200.txt', 'w')

for raw_name in os.listdir(data_folder_path):
    f.write(f'{os.path.join(data_folder_path, raw_name)}\n')
    
f.close()
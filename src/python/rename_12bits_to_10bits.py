import numpy as np
import os


data_folder_path = '/home/user/work/github/PMRID/results/20240812170511'

raw_paths = list()
for raw_name in os.listdir(data_folder_path):
    if not '.raw' in raw_name:
        continue
    raw_path = os.path.join(data_folder_path, raw_name)
    new_raw_path = raw_path.replace('RG12', 'RG10')
    os.rename(raw_path, new_raw_path)
    
    

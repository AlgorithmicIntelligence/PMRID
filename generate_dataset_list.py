import os 
import numpy as np
from argparse import ArgumentParser
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--folder_path', type=str, required=True)
    parser.add_argument('-p', '--prefix', type=str, default='')
    parser.add_argument('-t', '--type', type=str, default='train')
    args = parser.parse_args()
    return args

def parsing_files(txt_path=None, folder_path=None, type='sid'):
    iso_table = dict()
    ## type: 'vicore', 'sid'
    if type == 'sid':
        folder_path = os.path.dirname(txt_path)
        with open(txt_path, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            items = line.split()
            arw_file_name = items[1]
            iso = int(items[2][3:])
            
            raw_file_name = arw_file_name.replace('.ARW', '_h2848_w4256.raw')
            
            arw_file_path = os.path.join(folder_path, arw_file_name)
            raw_file_path = os.path.join(folder_path, 'raw', raw_file_name)
            if iso in iso_table:
                iso_table[iso].append(raw_file_path)
            else:
                iso_table[iso] = list(raw_file_path)
                        
    return iso_table
            
def generate_dataset_list(folder_path, prefix, type):
    list_folder = 'data'
    if not os.path.isdir(list_folder):
        os.mkdir(list_folder)
    prefix = prefix.split(',')
    
    list_path = os.path.join(list_folder, type) + '.txt'
    print(list_path)
    f = open(list_path, 'w')
    for data_name in os.listdir(folder_path):
        prefix_find = False
        for prefix_pattern in prefix:
            if data_name.startswith(prefix_pattern):
                prefix_find = True
                break
        if prefix_find == False:
            continue
        data_path = os.path.join(folder_path, data_name)
        f.write(data_path)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    args = get_args()
    
    generate_dataset_list(**vars(args))
    
    
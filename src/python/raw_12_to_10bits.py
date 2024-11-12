import os
import numpy as np

data_folder_path = '/home/user/work/data/Tsing/2-25600/linear/raws'

raw_paths = list()
for raw_name in os.listdir(data_folder_path):
    if not '.raw' in raw_name:
        continue
    raw_path = os.path.join(data_folder_path, raw_name)
    raw = np.fromfile(raw_path, np.uint16).astype(np.float32)
    os.remove(raw_path)
    # print(raw.shape)
    # print(raw.shape, raw.max(), raw.min(), raw.mean())
    raw = np.round(raw / 4095 * 1023).astype(np.uint16)
    # output_raw_path = raw_path.replace('h1080', 'h1072')
    with open(raw_path, 'wb') as f:
        f.write(raw.tobytes())

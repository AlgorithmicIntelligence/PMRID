import rawpy
import os

img_folder_path = '/home/user/work/data/SID/Sony/long'
result_folder_path = os.path.join(img_folder_path, 'raw')
if not os.path.isdir(result_folder_path):
    os.makedirs(result_folder_path)

for arw_name in os.listdir(img_folder_path):
    if not arw_name.endswith('.ARW'):
        continue
    arw_path = os.path.join(img_folder_path, arw_name)
    rawimg = rawpy.imread(arw_path).raw_image_visible
    h, w = rawimg.shape
    raw_name = arw_name.replace('.ARW', f'_h{h}_w{w}.raw')
    raw_path = os.path.join(result_folder_path, raw_name)
    with open(raw_path, 'wb') as f:
        f.write(rawimg.tobytes())

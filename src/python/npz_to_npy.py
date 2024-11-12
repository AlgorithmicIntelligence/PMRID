import numpy as np
import os
import argparse
import cv2

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str, default='/home/user/work/github/PMRID/data/valid/20240715164932.txt')

    args = parser.parse_args()
    return args

def get_dataset_txt(txt_path):
    npy_folder_path = os.path.dirname(txt_path)
    npy_folder_path = os.path.join(npy_folder_path, 'calibration')
    if not os.path.isdir(npy_folder_path):
        os.makedirs(npy_folder_path)
    output_dataset_txt = os.path.join(npy_folder_path, 'dataset.txt')
    f_out = open(output_dataset_txt, 'w')
        
    with open(txt_path, 'r') as f:
        data_paths = f.read().splitlines()
        
    num_images = len(data_paths)
    
    total_loss = 0
    inference_count = 0
    for data_path in data_paths:
        base_name = os.path.basename(data_path)
        base_name = os.path.splitext(base_name)[0]
        # print(f'VAL PATH: {data_path}')
        data = np.load(data_path)
        imgs = data['imgs']
        gt = data['gt']
        img_resize = list()
        gt_resize = list()
        img_resize.append(cv2.resize(imgs[0,0], (960, 544)))
        img_resize.append(cv2.resize(imgs[0,1], (960, 544)))
        img_resize.append(cv2.resize(imgs[0,2], (960, 544)))
        img_resize.append(cv2.resize(imgs[0,3], (960, 544)))
        gt_resize.append(cv2.resize(gt[0,0], (960, 544)))
        gt_resize.append(cv2.resize(gt[0,1], (960, 544)))
        gt_resize.append(cv2.resize(gt[0,2], (960, 544)))
        gt_resize.append(cv2.resize(gt[0,3], (960, 544)))
        img_resize = np.expand_dims(np.array(img_resize), axis=0)
        gt_resize = np.expand_dims(np.array(gt_resize), axis=0)
        
        print(f'{imgs.shape} {gt.shape}')
        print(f'dtype: {imgs.dtype} {img_resize.dtype}')
        print(f'shape: {imgs.shape} {img_resize.shape}')
        
        input_npy_path = os.path.join(npy_folder_path, base_name + "_input.npy")
        gt_npy_path = os.path.join(npy_folder_path, base_name + "_gt.npy")
        
        np.save(input_npy_path, img_resize)
        np.save(gt_npy_path, gt_resize)
        
        f_out.write(f'{input_npy_path}\n')
        f_out.write(f'{gt_npy_path}\n')
        
if __name__ == '__main__':
    args = load_args()
    get_dataset_txt(args.txt)
    
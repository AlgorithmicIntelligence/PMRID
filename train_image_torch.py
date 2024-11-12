#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import megengine as mge
import megengine.optimizer
import megengine.functional as F
from megengine.autodiff import GradManager
from src.python.data_processing import DataLoader, DataProcessor

from tqdm import tqdm
from loguru import logger

# from models.net_mge_org import Network, get_loss_l1
from basicsr.models.archs.CGNet_arch import CascadedGaze
from models.net_mge_light import Network, get_loss_l1
from dataset.training import CleanRawImages, DataAug, DataAugOptions, ImageIndexProcessor
from datetime import datetime
import time

img_channel = 4

width = 60
width = 8
enc_blks = [2, 2, 4, 6]
middle_blk_num = 10
dec_blks = [2, 2, 2, 2]

# width = 8
# enc_blks = [1, 1, 1, 1]
# middle_blk_num = 1
# dec_blks = [1, 1, 1, 1]

GCE_CONVS_nums = [3,3,2,2]


net = CascadedGaze(img_channel=img_channel,width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,GCE_CONVS_nums=GCE_CONVS_nums)


# python3 train_image.py --use-existed-valid --valid-txt data/valid/20240809171358.txt --pretrain checkpoints/20240812110617/epoch1502_iter162_trainingloss_0.004422_validloss_0.003808.pkl

t_info = datetime.now()
time_message = str(t_info.year) + \
                str(t_info.month).zfill(2) + \
                str(t_info.day).zfill(2) + \
                str(t_info.hour).zfill(2) + \
                str(t_info.minute).zfill(2) + \
                str(t_info.second).zfill(2)
                
def generate_validation_set(valid_txt:str, aug_obj:DataProcessor):
    valid_loader = DataLoader(valid_txt, mode='valid')
    # valid_loader = DataLoader(valid_txt)
    white_level = 16383
    black_level = 512
    save_folder_path = os.path.join('data/valid', time_message)
    valid_txt_path = os.path.join('data/valid', time_message) + '.txt'
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path)  
    
    f = open(valid_txt_path, 'w')
        
    for iter in tqdm(range(len(valid_loader)), dynamic_ncols=True):
        image_path, iso = valid_loader.image_infos[valid_loader.cur_image_index]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        imgs, g_means = valid_loader.get_samples(sample_size=1)
        imgs_noisy, imgs_gt, cvt_k, cvt_b = aug_obj.transform(imgs, g_means)
        valid_data_path = os.path.join(save_folder_path, f'{image_name}.npz')
        np.savez(valid_data_path, imgs=imgs_noisy, gt=imgs_gt, norm_k=cvt_k)
        f.write(valid_data_path)
        f.write('\n')
    f.close()
    return valid_txt_path
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-aug-config', type=Path)
    parser.add_argument('--train-txt', type=Path, default='/home/user/work/data/SID/Sony_train_list_raw.txt')
    parser.add_argument('--valid-txt', type=Path, default='/home/user/work/data/SID/Sony_val_list_raw.txt')
    # parser.add_argument('--valid-txt', type=Path, default='/home/user/work/github/PMRID/data/valid/20240417154530.txt')
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ckp-dir', default=Path('./checkpoints'), type=Path)
    # parser.add_argument('--pretrain', default='./checkpoints/20240417110728/epoch_161000_loss_0.031864.pkl', type=str)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--learning-rate', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=8000, type=int)
    parser.add_argument('--use-existed-valid', action='store_true')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    
    ckp_dir = os.path.join(args.ckp_dir, time_message)    
    if not os.path.isdir(ckp_dir):
        os.makedirs(ckp_dir)
        

    # Configure loggger
    logger.configure(handlers=[dict(
        sink=lambda msg: tqdm.write(msg, end=''),
        format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] [<level>{level}</level>] {message}",
        colorize=True
    )])
    
        
    # Create model
    # net = Network()
    net = CascadedGaze(img_channel=img_channel,width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,GCE_CONVS_nums=GCE_CONVS_nums)

    
    global_step = 0
    init_epoch = 0
    if args.pretrain:
        net.load_state_dict(megengine.load(args.pretrain))
        init_epoch = int(args.pretrain.split('epoch')[1].split('_')[0])
        global_step = init_epoch * int(args.pretrain.split('iter')[1].split('_')[0])
    # Create optimizer
    optimizer = megengine.optimizer.Adam(net.parameters(), lr=args.lr)
    # Create GradManager
    gm = GradManager().attach(net.parameters())
    # aug_opts = DataAugOptions.parse_file(args.data_aug_config)

    noise_k = (0.0005995267, 0.00868861)
    noise_b = (7.11772e-7, 6.514934e-4, 0.11492713)
    train_aug = DataProcessor(noise_k, noise_b)
    if args.use_existed_valid == False:
        valid_txt = generate_validation_set(args.valid_txt, train_aug) 
    else:
        valid_txt = args.valid_txt
        
    train_ds = DataLoader(args.train_txt)
    train_loader = train_ds
    
    # train_ds = CleanRawImages(data_txt=args.train_txt, opts=aug_opts)
    # train_loader = DataLoader(train_ds, batch_size=batch_size)

    # learning rate scheduler
    def adjust_learning_rate(opt, epoch, step):
        M = len(train_ds) // args.batch_size
        T = M * 100
        Th = T // 2

        # # warm up
        # if base_lr > 2e-3 and step < T:
        #     return 1e-4

        if epoch < 3000:
            f = 1 - step / (M*3000)
        elif epoch < 3000:
            f = 0.1
        elif epoch < 5000:
            f = 0.2
        else:
            f = 0.1

        t = step % T
        if t < Th:
            f2 = t / Th
        else:
            f2 = 2 - (t/Th)

        lr = f * f2 * args.lr

        for pgroup in opt.param_groups:
            pgroup["lr"] = lr

        return lr

    # train step
    def train_step(img, gt, norm_k):
        with gm:
            pred = net(img)
            loss = get_loss_l1(pred, gt, norm_k)
            gm.backward(loss)
        optimizer.step().clear_grad()
        return loss
    
    def val_step(valid_txt):
        patch_size = 13
        patch_radius = 6
        height = 2848
        width = 4256
        batch_size = 2**13
        net.eval()
        with open(valid_txt, 'r') as f:
            data_paths = f.read().splitlines()
            
        num_images = len(data_paths)
        
        total_loss = 0
        inference_count = 0
        for data_path in tqdm(data_paths, dynamic_ncols=True):
            # print(f'VAL PATH: {data_path}')
            data = np.load(data_path)
            imgs = data['imgs']
            gt = data['gt']
            norm_k = data['norm_k']
            
            b, c, h, w = gt.shape
            img = mge.tensor(imgs)
            gt = mge.tensor(gt) 
            norm_k = mge.tensor(norm_k) 
            pred = net(img)
            # print(f"VAL INPUT: {img.shape} {img.max()} {img.min()} {img.mean()}")
            # print(f"VAL GOLDEN: {gt.shape} {gt.max()} {gt.min()} {gt.mean()}")
            # print(f"VAL PRED: {pred.shape} {pred.max().item()} {pred.min().item()} {pred.mean().item()}")
            loss = get_loss_l1(pred, gt, norm_k)
            total_loss += loss
            inference_count += 1
        total_loss /= inference_count
        net.train()
        return total_loss.item()
                        

    # train loop
    best_loss = float('inf')
    for epoch in range(init_epoch, args.num_epoch):
        train_ds.reset_index_and_shuffle_image_infos()
        train_loss = 0
        num_samples = 0
                
        # while train_loader.is_samples_remaining():
        #     imgs, g_means = train_loader.get_samples(sample_size=batch_size)
        
        for iter in tqdm(range(len(train_loader)//batch_size), dynamic_ncols=True):
            imgs, g_means = train_loader.get_samples(sample_size=batch_size)
            imgs, gt, norm_k, cvt_b = train_aug.transform(imgs, g_means)
            lr = adjust_learning_rate(optimizer, epoch, global_step)
            t4 = time.time()
            loss = train_step(imgs, gt, norm_k)
            t5 = time.time()
            train_loss += loss
            cur_loss = train_loss.item()/(iter+1)
            num_samples += len(imgs)
            global_step += 1
            t6 = time.time()
        val_loss = val_step(valid_txt)
        mge.save(net.state_dict(), os.path.join(ckp_dir, f"epoch{epoch+1}_iter{iter+1}_trainingloss_{cur_loss:.6f}_validloss_{val_loss:.6f}.pkl"))
        logger.info(f"epoch: {epoch+1}, train_loss: {cur_loss:.6f}, valid_loss: {val_loss:.6f}")
        # if eval_loss < best_loss:
        #     mge.save(net.state_dict(), os.path.join(ckp_dir, f"epoch_{epoch+1}_loss_{eval_loss}.pkl"))
        #     best_loss = eval_loss
        #     print(f'save best loss: {best_loss} in {os.path.join(ckp_dir, f"epoch_{epoch+1}_loss_{eval_loss}.pkl")}')
            # mge.save(net.state_dict(), "test.pkl")

if __name__ == "__main__":
    

    main()
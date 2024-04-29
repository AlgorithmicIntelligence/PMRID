#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import megengine as mge
import megengine.optimizer
import megengine.functional as F
from megengine.data import DataLoader, RandomSampler
from megengine.autodiff import GradManager

from tqdm import tqdm
from loguru import logger

from models.net_mge import Network, get_loss_l1
from dataset.training import CleanRawImages, DataAug, DataAugOptions, ImageIndexProcessor
from datetime import datetime
import time

t_info = datetime.now()
time_message = str(t_info.year) + \
                str(t_info.month).zfill(2) + \
                str(t_info.day).zfill(2) + \
                str(t_info.hour).zfill(2) + \
                str(t_info.minute).zfill(2) + \
                str(t_info.second).zfill(2)
                
def generate_validation_set(valid_txt:str, aug_obj:DataAug):
    padding_radius = 6
    white_level = 16383
    black_level = 512
    save_folder_path = os.path.join('data/valid', time_message)
    valid_txt_path = os.path.join('data/valid', time_message) + '.txt'
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path)  
    
    # valid_ds = CleanRawImages(data_txt=valid_txt, opts=aug_obj)
    # valid_loader = ImageIndexProcessor(valid_txt)
    with open(valid_txt, 'r') as f:
        image_infos = [line.split() for line in f.read().splitlines()]    
            # print(f' input shape: {imgs.shape}, g_means shape: {g_means.shape}')
    # valid_loader = DataLoader(valid_ds, sampler=RandomSampler(valid_ds, batch_size=1, drop_last=True))
    f = open(valid_txt_path, 'w')
    for image_info in image_infos:
        image_path, iso = image_info
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        rawimg = np.fromfile(image_path, np.uint16).reshape(2848, 4256)
        rawimg = np.pad(rawimg, ((padding_radius, padding_radius), (padding_radius, padding_radius)), mode='reflect')
        rawimg = (rawimg - black_level) / (white_level - black_level)
        raw_mean = rawimg.mean()
        rawimg = np.expand_dims(rawimg, axis=(0,-1))
        raw_mean = np.expand_dims(raw_mean, axis=0)
    # for iter in tqdm(range(len(valid_loader)//batch_size), dynamic_ncols=True):
    
        # imgs, g_means = valid_loader.get_samples(sample_size=batch_size)
        imgs, gt, norm_k = aug_obj.transform(rawimg, raw_mean, mode='general')
        gt = gt[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        
        valid_data_path = os.path.join(save_folder_path, f'{image_name}.npz')
        np.savez(valid_data_path, imgs=imgs, gt=gt, norm_k=norm_k)
        f.write(valid_data_path)
        f.write('\n')
    f.close()
    return valid_txt_path
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-aug-config', type=Path)
    parser.add_argument('--train-txt', type=Path, default='/home/user/work/data/SID/Sony_train_list_raw.txt')
    # parser.add_argument('--valid-txt', type=Path, default='/home/user/work/data/SID/Sony_val_list_raw.txt')
    parser.add_argument('--valid-txt', type=Path, default='/home/user/work/github/PMRID/data/valid/20240417154530.txt')
    parser.add_argument('--batch-size', default=2**13, type=int)
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
    net = Network()
    if args.pretrain:
        net.load_state_dict(megengine.load(args.pretrain))
    # Create optimizer
    optimizer = megengine.optimizer.Adam(net.parameters(), lr=args.lr)
    # Create GradManager
    gm = GradManager().attach(net.parameters())
    # aug_opts = DataAugOptions.parse_file(args.data_aug_config)
    aug_opts = DataAugOptions()
    train_aug = DataAug(aug_opts)
    if args.use_existed_valid == False:
        valid_txt = generate_validation_set(args.valid_txt, train_aug) 
    else:
        valid_txt = args.valid_txt
        
    train_ds = ImageIndexProcessor(args.train_txt)
    train_loader = train_ds
    
    # train_ds = CleanRawImages(data_txt=args.train_txt, opts=aug_opts)
    # train_loader = DataLoader(train_ds, batch_size=batch_size)

    # learning rate scheduler
    def adjust_learning_rate(opt, epoch, step):
        num_training_samples = 2848*4256*162
        # M = len(train_ds) // args.batch_size
        M = num_training_samples // args.batch_size
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
        # net.eval()
        with open(valid_txt, 'r') as f:
            data_paths = f.read().splitlines()
            
        num_images = len(data_paths)
        
        total_loss = 0
        inference_count = 0
        for data_path in tqdm(data_paths, dynamic_ncols=True):
            data = np.load(data_path)
            imgs = data['imgs']
            gt = data['gt']
            norm_k = data['norm_k']
            
            b, c, h, w = gt.shape
            batch_input = list()
            batch_golden = list()
            batch_k = list()
            for pixel_index in tqdm(range(h*w), dynamic_ncols=True):
                y = pixel_index // width
                x = pixel_index % width
                
                input = imgs[0, :, y:y+patch_size, x:x+patch_size]
                golden = gt[0, :, y:y+1, x:x+1]
                batch_input.append(input)
                batch_golden.append(golden)
                batch_k.append(norm_k[0])
                
                if len(batch_input) == batch_size or pixel_index == h*w-1:
                    batch_input = mge.tensor(np.stack(batch_input))
                    batch_golden = mge.tensor(np.stack(batch_golden)) 
                    batch_k = mge.tensor(np.stack(batch_k)) 
                    pred = net(batch_input)
                    loss = get_loss_l1(pred, batch_golden, batch_k)
                    total_loss += loss
                    inference_count += 1
                    batch_input = list()
                    batch_golden = list()  
                    batch_k = list()
        total_loss /= inference_count
        net.train()
        return total_loss.item()
                        

    # train loop
    best_loss = float('inf')
    global_step = 0
    for epoch in range(args.num_epoch):
        train_ds.reset_index_and_shuffle_image_infos()
        train_loss = 0
        num_samples = 0
                
        # while train_loader.is_samples_remaining():
        #     imgs, g_means = train_loader.get_samples(sample_size=batch_size)
        
        for iter in tqdm(range(len(train_loader)//batch_size), dynamic_ncols=True):
            t1 = time.time()
            imgs, g_means = train_loader.get_samples(sample_size=batch_size)
            t2 = time.time()
            # print(f' input shape: {imgs.shape}, g_means shape: {g_means.shape}')
            imgs, gt, norm_k = train_aug.transform(imgs, g_means)
            t3 = time.time()
            lr = adjust_learning_rate(optimizer, epoch, global_step)
            # imgs *= 256
            # gt *= 256
            t4 = time.time()
            loss = train_step(imgs, gt, norm_k)
            t5 = time.time()
            train_loss += loss
            cur_loss = train_loss.item()/(iter+1)
            num_samples += len(imgs)
            global_step += 1
            t6 = time.time()
            # logger.info(f"Time - get_samples: {t2-t1:.3f}, transform: {t3-t2:.3f}, train: {t5-t4:.3f}, total: {t6-t1:.3f}")
            if (iter+1) % 50 == 0:
                logger.info(f"epoch: {epoch+1}, iter: {iter+1}, train_loss: {cur_loss:.6f}, cur_loss: {loss.item():.6f}")
            if (iter+1) % 1000 == 0:
                val_loss = val_step(valid_txt)
                logger.info(f"epoch: {epoch+1}, iter: {iter+1}, train_loss: {cur_loss:.6f}, valid_loss: {val_loss:.6f}")
                mge.save(net.state_dict(), os.path.join(ckp_dir, f"epoch{epoch+1}_iter{iter+1}_trainingloss_{cur_loss:.6f}_validloss_{val_loss:.6f}.pkl"))
            if cur_loss < best_loss:
                best_loss = cur_loss
                mge.save(net.state_dict(), os.path.join(ckp_dir, f"epoch{epoch+1}_iter{iter+1}_trainingloss_{cur_loss:.6f}.pkl"))
        
        # logger.info(f"epoch: {epoch+1}, train_loss: {train_loss}, eval_loss: {eval_loss}, lr: {lr}")
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
#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.python.data_processing_tf import DataLoader, DataProcessor

from tqdm import tqdm
from loguru import logger
from datetime import datetime

# 將這些函數替換成用 TensorFlow 定義的版本
from models.net_tf import Network, get_loss_l1

# TODO: change CascadedNet
# from models.net_tf import Network, get_loss_l1

t_info = datetime.now()
time_message = str(t_info.year) + \
                str(t_info.month).zfill(2) + \
                str(t_info.day).zfill(2) + \
                str(t_info.hour).zfill(2) + \
                str(t_info.minute).zfill(2) + \
                str(t_info.second).zfill(2)
                
def generate_validation_set(valid_txt:str, aug_obj:DataProcessor):
    valid_loader = DataLoader(valid_txt, mode='valid')
    save_folder_path = os.path.join('data/valid', time_message)
    valid_txt_path = os.path.join('data/valid', time_message) + '.txt'
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path)  
    
    f = open(valid_txt_path, 'w')
        
    for iter in tqdm(range(len(valid_loader)), dynamic_ncols=True):
        image_path, iso = valid_loader.image_infos[valid_loader.cur_image_index]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        imgs, g_means = valid_loader.get_samples(sample_size=1)
        # print('1')
        imgs_noisy, imgs_gt, cvt_k, cvt_b = aug_obj.transform(imgs, g_means)
        # print('####2####')
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
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--ckp-dir', default=Path('./checkpoints'), type=Path)
    parser.add_argument('--pretrain', type=str)
    parser.add_argument('--learning-rate', dest='lr', default=1e-3, type=float)
    parser.add_argument('--num-epoch', default=8000, type=int)
    parser.add_argument('--use-existed-valid', action='store_true')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    
    ckp_dir = os.path.join(args.ckp_dir, time_message)    
    if not os.path.isdir(ckp_dir):
        os.makedirs(ckp_dir)
        

    # Configure logger
    logger.configure(handlers=[dict(
        sink=lambda msg: tqdm.write(msg, end=''),
        format="[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] [<level>{level}</level>] {message}",
        colorize=True
    )])
    
        
    # Create model
    net = Network()
    net.build(input_shape=(None, 512, 512, 4))
    global_step = 0
    init_epoch = 0
    if args.pretrain:
        net.load_weights(args.pretrain)
        # 假設有相應的代碼來解析epoch和step
        init_epoch = int(args.pretrain.split('epoch')[1].split('_')[0])
        global_step = init_epoch * int(args.pretrain.split('iter')[1].split('_')[0])
        
    # Create optimizer
    optimizer = Adam(learning_rate=args.lr)
    
    noise_k = (0.0005995267, 0.00868861)
    noise_b = (7.11772e-7, 6.514934e-4, 0.11492713)
    train_aug = DataProcessor(noise_k, noise_b)
    if not args.use_existed_valid:
        valid_txt = generate_validation_set(args.valid_txt, train_aug) 
    else:
        valid_txt = args.valid_txt
    print('################ finish generate validationset ##################')
        
    train_ds = DataLoader(args.train_txt)

    # learning rate scheduler
    def adjust_learning_rate(opt, epoch, step):
        M = len(train_ds) // args.batch_size
        T = M * 100
        Th = T // 2

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
        opt.learning_rate = lr

        return lr

    # train step
    @tf.function
    def train_step(img, gt, norm_k):
        with tf.GradientTape() as tape:
            pred = net(img, training=True)
            loss = get_loss_l1(pred, gt, norm_k)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return loss
    
    def val_step(valid_txt):
        # net.eval()
        total_loss = 0
        inference_count = 0
        with open(valid_txt, 'r') as f:
            data_paths = f.read().splitlines()
        
        for data_path in tqdm(data_paths, dynamic_ncols=True):
            data = np.load(data_path)
            imgs = data['imgs']
            gt = data['gt']
            norm_k = data['norm_k']
            
            pred = net(imgs, training=False)
            loss = get_loss_l1(pred, gt, norm_k)
            total_loss += loss.numpy()
            inference_count += 1
        total_loss /= inference_count
        return total_loss
                        
    # train loop
    print('################ start training ##################')
    for epoch in range(init_epoch, args.num_epoch):
        train_ds.reset_index_and_shuffle_image_infos()
        train_loss = 0
        num_samples = 0
                
        for iter in tqdm(range(len(train_ds)//batch_size), dynamic_ncols=True):
            imgs, g_means = train_ds.get_samples(sample_size=batch_size)
            # print('imgshape:' ,imgs.shape)
            imgs, gt, norm_k, cvt_b = train_aug.transform(imgs, g_means)
            # print('imgshape:' ,imgs.shape)
            lr = adjust_learning_rate(optimizer, epoch, global_step)
            loss = train_step(imgs, gt, norm_k)
            train_loss += loss
            cur_loss = train_loss/(iter+1)
            num_samples += len(imgs)
            global_step += 1
        
        val_loss = val_step(valid_txt)
        net.save_weights(os.path.join(ckp_dir, f"epoch{epoch+1}_iter{iter+1}_trainingloss_{cur_loss:.6f}_validloss_{val_loss:.6f}.h5"))
        logger.info(f"epoch: {epoch+1}, train_loss: {cur_loss:.6f}, valid_loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()

import os, json
import math
from enum import Enum
from typing import Optional, List, Tuple
import rawpy
import numpy as np
import cv2
from numba import jit, njit, types
from numba.typed import List
import numba

from pydantic import BaseModel
from megfile import SmartPath, smart_load_from
import random
import time
from loguru import logger
import tensorflow as tf


class BayerPattern(Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


class RawImageItem(BaseModel):
    path: str
    width: int
    height: int
    black_level: int = 512
    # white_level: int = 65535
    white_level: int = 16383
    bayer_pattern: BayerPattern = BayerPattern.RGGB
    g_mean_01: float


class NoiseProfile(BaseModel):
    K: Tuple[float, float] = (0.0005995267, 0.00868861)
    B: Tuple[float, float, float] = (7.11772e-7, 6.514934e-4, 0.11492713)
    value_scale: float = 959.0


class DataAugOptions(BaseModel):
    noise_profile: NoiseProfile = NoiseProfile()
    camera_value_scale: float = 959.0
    iso_range: Tuple[float, float] = (100, 6400)
    anchor_iso: float = 1600.0
    output_shape: Tuple[int, int] = (512, 512)   # 512x512x4
    output_shape: Tuple[int, int] = (1024, 1024)   # 512x512x4
    target_brighness_range: Tuple[float, float] = (0.02, 0.5)


@njit
def get_image_shape(image_path):
    # Load image and generate index list
    image = rawimg = rawpy.imread(image_path).raw_image_visible
    height, width = image.shape
    return height, width        
    
# @jit(nopython=True)
# @njit
def get_next_block_image_index_list(image_infos, cur_block_index, number_of_blocks, block_size):
    block_index_list = list()
    if cur_block_index == number_of_blocks:
        return
    cur_image_infos_block = image_infos[cur_block_index * block_size: (cur_block_index+1) * block_size]
    for image_idx, image_info in enumerate(cur_image_infos_block):
        print(f'image idx in block: {image_idx}')
        image_path = image_info[0]
        # iso = int(str(image_info[1][3:]))
        iso = int(0)
        
        
        h, w = 2848, 4256
        # h, w = ImageIndexProcessor.get_image_shape(image_path)
        for y in range(h):
            for x in range(w):
                block_index_list.append((image_path, iso, y, x))
    return block_index_list

class ValidationDataloader():
    def __init__(self, image_info_file, block_size=1):
        self.image_info_file = image_info_file
        self.block_size = block_size
        
        # Read image info from file
        self.image_infos = self._read_image_info()
        # self.reset_index_and_shuffle_image_infos()
        
        self.num_images = len(self.image_infos)
        self.number_of_blocks = math.ceil(self.num_images / self.block_size)
        self.cur_image_index = 0    
        self.cur_x_index = 0
        self.cur_y_index = 0    
        self.block_index_list = list()
        
    def _read_image_info(self):
        with open(self.image_info_file, 'r') as f:
            image_infos = [line.split() for line in f.read().splitlines()]    
        return image_infos    
    
    def get_samples(self, sample_size=1):
        
        if sample_size > len(self.block_index_list):
            next_block_index_list = get_next_block_image_index_list(List(self.image_infos), self.cur_block_index, self.number_of_blocks, self.block_size)
            self.block_index_list += next_block_index_list
            self.cur_block_index += 1
        samples_info = self.block_index_list[:sample_size]
        self.block_index_list = self.block_index_list[sample_size:]
        samples, samples_g_mean = self._process_samples(samples_info)
 
        return samples, samples_g_mean
    def _process_samples(self, samples_info, black_level=512, white_level=16383, patch_size=13):
        assert patch_size % 2 
        
        patch_radius = (patch_size -1) // 2
        images = list()
        images_g_mean = list()
        for sample_info in samples_info:
            image_path, iso, y, x = sample_info
            rawimg = rawpy.imread(image_path).raw_image_visible
            rawimg = self.reflect_padding(rawimg, patch_radius)
            rawimg = self.crop_and_random_flip(rawimg, x+patch_radius, y+patch_radius, crop_radius=patch_radius)

            raw01 = (rawimg - black_level) / (white_level - black_level)
            g_mean_01 = raw01.mean()
            # H, W = raw01.shape
            # pixel shuffle to RGGB image
            # rggb01 = raw01.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
            images.append(raw01)
            images_g_mean.append(g_mean_01)
        images = np.stack(images)
        
        return images, np.array(images_g_mean)
    

class DataLoader():
    def __init__(self, image_info_file, image_shape=(2848,4256), mode='train'):
        self.image_info_file = image_info_file
        self.mode = mode
        
        # Read image info from file
        self.image_infos = self._read_image_info()
        self.image_height, self.image_width = image_shape
        self.num_images = len(self.image_infos)
        self.cur_image_index = 0
        
    def _read_image_info(self):
        with open(self.image_info_file, 'r') as f:
            image_infos = [line.split() for line in f.read().splitlines()]    
        return image_infos    
    
    def reset_index_and_shuffle_image_infos(self):
        self.cur_image_index = 0
        random.shuffle(self.image_infos)

    def get_samples(self, sample_size=1):
        # input_batch = list()
        for _ in range(sample_size):
            sample_info = self.image_infos[self.cur_image_index]
            self.cur_image_index += 1
            samples, samples_g_mean = self._process_samples(sample_info)    
        return samples, samples_g_mean
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self):
        return self.get_samples(1)
         
    def _process_samples(self, sample_info, num_patches=8, black_level=512, white_level=16383, output_shape=(512, 512)):
        image_path, iso = sample_info
        h, w = self.image_height, self.image_width
        oh, ow = output_shape

        images = list()
        images_g_mean = list()
        
        rawimg = np.fromfile(image_path, np.uint16).reshape(h, w)
        rawimg = rawimg.astype(np.float32)
        rawimg = (rawimg - black_level) / (white_level - black_level)
        if self.mode != 'train':
            num_patches = 1
        
        for _ in range(num_patches):
            if self.mode == 'train':
                raw_crop = self.random_flip_and_crop(rawimg, output_shape=output_shape)  
            else:
                raw_crop = rawimg 
                     
            images.append(raw_crop)
            # images_g_mean.append(g_mean)
        images = np.stack(images)
        # images_g_mean = np.stack(images_g_mean)
        oh, ow = images.shape[1:]
        rggb = images.reshape(-1, oh//2, 2, ow//2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, oh//2, ow//2, 4)
        rggb_mean = rggb.mean(axis=(1, 2))
        images_g_mean = rggb_mean[:, 1:3].mean(axis=1)
        return rggb, images_g_mean
        
    def random_flip_and_crop(self, img: np.ndarray, src_bayer_pattern: BayerPattern=BayerPattern.BGGR, output_shape=(512, 512)) -> np.ndarray:
        """
        Random flip and crop a bayter-patterned image, and normalize the bayer pattern to RGGB.
        """

        flip_ud = np.random.rand() > 0.5
        flip_lr = np.random.rand() > 0.5

        if src_bayer_pattern == BayerPattern.RGGB:
            crop_x_offset, crop_y_offset = 0, 0
        elif src_bayer_pattern == BayerPattern.GBRG:
            crop_x_offset, crop_y_offset = 0, 1
        elif src_bayer_pattern == BayerPattern.GRBG:
            crop_x_offset, crop_y_offset = 1, 0
        elif src_bayer_pattern == BayerPattern.BGGR:
            crop_x_offset, crop_y_offset = 1, 1

        if flip_lr:
            crop_x_offset = (crop_x_offset + 1) % 2
        if flip_ud:
            crop_y_offset = (crop_y_offset + 1) % 2

        h, w = img.shape
        ho, wo = output_shape

        x0, y0 = np.random.randint(0, w - wo), np.random.randint(0, h - ho)
        x0, y0 = x0 // 2 * 2 + crop_x_offset, y0 // 2 * 2 + crop_y_offset

        img_crop = img[y0:y0+ho, x0:x0+wo]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)

        return img_crop


class DataProcessor:
    def __init__(self, noise_k: Tuple[float, float], noise_b: Tuple[float, float, float]):
        self.poly_k = np.poly1d(noise_k)
        self.poly_b = np.poly1d(noise_b)
        
        self.target_brighness_range: Tuple[float, float] = (0.02, 0.5)
        self.anchor_iso: float = 1600.0
        self.iso_range: Tuple[float, float] = (100, 6400)
        self.value_scale = 959
        self.debug_cnt = 0
        
        
    def transform(self, imgs, images_g_mean, value_scale=959.0, channel_first=False, isos=None, mode='train'):
        # change into channel first
        if channel_first:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        imgs = imgs * value_scale      
        if mode == 'train': 
            imgs_gt = self.brightness_augmentation(imgs, images_g_mean)
            imgs_noisy, isos = self.add_noise(imgs_gt)
        else:
            imgs_gt = imgs
            imgs_noisy = imgs
        # black_level=512
        # white_level=16383
        # debug_imgs_noisy = imgs_noisy/value_scale * (white_level-black_level) + black_level
        # debug_imgs_gt = imgs_gt/value_scale * (white_level-black_level) + black_level
        # with open(f'{self.debug_cnt}_noisy_r.raw', 'wb') as f:
        #     f.write(np.array(debug_imgs_noisy[0, 0]).astype(np.uint16).tobytes())
        # with open(f'{self.debug_cnt}_gt_r.raw', 'wb') as f:
        #     f.write(np.array(debug_imgs_gt[0, 0]).astype(np.uint16).tobytes())
        self.debug_cnt += 1
        imgs_gt, cvt_k, cvt_b = self.k_sigma_transform(imgs_gt, isos)
        imgs_noisy, cvt_k, cvt_b = self.k_sigma_transform(imgs_noisy, isos)
        imgs_noisy /= value_scale
        imgs_gt /= value_scale
        return imgs_noisy, imgs_gt, cvt_k, cvt_b
    
    def add_noise(self, img: tf.Tensor, isos=None) -> Tuple[tf.Tensor, float]:
        """
        Args:
            - img: [-black, camera_value_scale]

        Returns:
            - noisy_img
            - iso
        """

        N = img.shape[0]
        if isos == None:
            isos = np.random.uniform(*self.iso_range, size=(N, ))
        k, b = self.noise_func(isos)
        k = k.reshape(-1, 1, 1, 1).astype(np.float32)
        b = b.reshape(-1, 1, 1, 1).astype(np.float32)        
        # print(f'IMGSHAPE: {img.shape}, KSHAPE: {k.shape}, {img.max().item()} {img.min().item()} {k.min().item()}, {k.max().item()}')

        # print(f'IMGSHAPE: {img.shape}, KSHAPE: {k.shape}, imgdtype: {img.dtype}, kdtype: {k.dtype}')
        # print(f'noise: {img.shape}')
        shot_noisy = tf.random.poisson(lam=img / k, shape=[1]) * k
        shot_noisy = shot_noisy[0]
        read_noisy = tf.random.normal(shape=img.shape) * np.sqrt(b)
        noisy = shot_noisy + read_noisy
        # noisy = read_noisy
        noisy = tf.round(noisy)
        return noisy, isos
                
    def brightness_augmentation(self, img_batch: tf.Tensor, images_g_mean: List[float]) -> tf.Tensor:
        low, high = self.target_brighness_range
        N = len(images_g_mean)
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N, )))
        s = np.clip(btarget / images_g_mean, 0.01, 1.0, dtype=np.float32)
        # print(f'brightness - orig_gmean: {images_g_mean}, img_batch_dtype: {img_batch.dtype}, s.dtype: {s.dtype}')
        return img_batch * s.reshape(-1, 1, 1, 1)
    
    def noise_func(self, iso):
        k = self.poly_k(iso)
        b = self.poly_b(iso)
        return k, b   
    
    def k_sigma(self, isos: float, anchor_iso) -> Tuple[float, float]:
        k, sigma = self.noise_func(isos)
        k_a, sigma_a = self.noise_func(anchor_iso)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        return cvt_k, cvt_b    
        
    def k_sigma_transform(self, imgs, isos):    

        
        cvt_k, cvt_b = self.k_sigma(isos, self.anchor_iso)
        cvt_k = cvt_k.astype(np.float32)
        cvt_b = cvt_b.astype(np.float32)

        imgs = imgs * cvt_k.reshape(-1, 1, 1, 1) + cvt_b.reshape(-1, 1, 1, 1)
        return imgs, cvt_k, cvt_b
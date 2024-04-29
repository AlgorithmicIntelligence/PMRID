import os, json
import math
from enum import Enum
from typing import Optional, List, Tuple
import rawpy
import numpy as np
import megengine as mge
import megengine.random
import megengine.functional as F
import cv2
from numba import jit, njit, types
from numba.typed import List
import numba

from pydantic import BaseModel
from megfile import SmartPath, smart_load_from
from megengine.data.dataset import Dataset
import random
import time
from loguru import logger


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

class ImageIndexProcessor():
    def __init__(self, image_info_file, block_size=2, mode='train'):
        self.image_info_file = image_info_file
        self.block_size = block_size
        self.mode = mode
        
        # Read image info from file
        self.image_infos = self._read_image_info()
        self.image_height = 2848
        self.image_width = 4256
        # self.reset_index_and_shuffle_image_infos()
        
        self.num_images = len(self.image_infos)
        self.number_of_blocks = math.ceil(self.num_images / self.block_size)
        self.cur_block_index = 0        
        self.block_index_list = list()
        
        if self.mode == 'valid':
            self.cur_image_index = 0
            self.cur_pixel_index = 0
        
    def _read_image_info(self):
        with open(self.image_info_file, 'r') as f:
            image_infos = [line.split() for line in f.read().splitlines()]    
        return image_infos    
    
    def reset_index_and_shuffle_image_infos(self):
        self.cur_image_index = 0
        self.cur_pixel_index = 0
        self.cur_block_index = 0      
        random.shuffle(self.image_infos)
    
    def is_samples_remaining(self):
        if len(self.block_index_list) or self.cur_block_index != self.number_of_blocks:
            return True
        return False

    def get_samples(self, sample_size=1):
        # if sample_size > len(self.block_index_list):
        #     print("############## GET NEXT BLOCK!!! ##############")
        #     next_block_index_list = get_next_block_image_index_list(List(self.image_infos), self.cur_block_index, self.number_of_blocks, self.block_size)
        #     self.block_index_list += next_block_index_list
        #     self.cur_block_index += 1
        if self.mode == 'train':
            sample_idx = random.randint(0, self.num_images-1)
            sample_info = self.image_infos[sample_idx]
        else:
            # sample_size = np.clip(sample_size, 0, self.image_height * self.image_width - self.cur_pixel_index)
            sample_info = self.image_infos[self.cur_image_index]
                
        samples, samples_g_mean = self._process_samples(sample_info, sample_size)        
        # t1 = time.time()
        # samples_info = self.block_index_list[:sample_size]
        # t2 = time.time()
        # self.block_index_list = self.block_index_list[sample_size:]
        # t3 = time.time()
        
        # t4 = time.time()
        # logger.info(f"Time - t12: {t2-t1:.3f}, t23: {t3-t2:.3f}, t34: {t4-t3:.3f}")
        return samples, samples_g_mean
    
    def __len__(self):
        return self.num_images * self.image_height * self.image_width
    
    def __getitem__(self):
        return self.get_samples(1)
         
    @staticmethod        
    @jit
    def get_image_shape(image_path):
        # Load image and generate index list
        image = rawimg = rawpy.imread(image_path).raw_image_visible
        height, width = image.shape
        return height, width        
    
    def shuffle_block_index_list(self):
        random.shuffle(self.block_index_list)       
                                    
    def _process_samples(self, sample_info, batch_size, black_level=512, white_level=16383, patch_size=13):
        assert patch_size % 2 
        image_path, iso = sample_info
        
        patch_radius = (patch_size -1) // 2
        images = list()
        images_g_mean = list()
        
        
            # raw_folder_name = os.path.join(os.path.dirname(image_path), 'raw')
            # raw_file_name = os.path.basename(image_path).replace('.ARW', '_h2848_w4256.raw')
            # raw_path = os.path.join(raw_folder_name, raw_file_name)
            # rawimg = rawpy.imread(image_path).raw_image_visible
        rawimg = np.fromfile(image_path, np.uint16).reshape(2848, 4256)
        rawimg = self.reflect_padding(rawimg, patch_radius)
        rawimg = (rawimg - black_level) / (white_level - black_level)
        
        for _ in range(batch_size):
            if self.mode == 'train':
                x = random.randint(0, self.image_width-1)
                y = random.randint(0, self.image_height-1)
            else:
                x = self.cur_pixel_index % self.image_width
                y = self.cur_pixel_index // self.image_width
                self.cur_pixel_index += 1
            raw_crop = self.crop_and_random_flip(rawimg, x+patch_radius, y+patch_radius, crop_radius=patch_radius)
            # raw01 = (rawimg - black_level) / (white_level - black_level)
            g_mean_01 = raw_crop.mean()
            # H, W = raw01.shape
            # pixel shuffle to RGGB image
            # rggb01 = raw01.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
            images.append(raw_crop)
            images_g_mean.append(g_mean_01)
            if self.mode == 'valid' and self.cur_pixel_index == self.image_height * self.image_width:
                self.cur_image_index += 1
                self.cur_pixel_index = 0
                image_path, iso = self.image_infos[self.cur_image_index]
                rawimg = np.fromfile(image_path, np.uint16).reshape(2848, 4256)
                rawimg = self.reflect_padding(rawimg, patch_radius)
                rawimg = (rawimg - black_level) / (white_level - black_level)
        # t5 = time.time()
        images = np.stack(images)
        # t6 = time.time()
        # logger.info(f"Time - ARW: {t2-t1:.3f}, RAW: {t21-t2:.3f}, t23: {t3-t21:.3f}, t34: {t4-t3:.3f}, t45: {t5-t4:.3f}, t56: {t6-t5:.3f}")
        
        return images, np.array(images_g_mean)
           
    def reflect_padding(self, img, padding_size):
        padded_image = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), mode='reflect')
        return padded_image
     
    def crop_and_random_flip(self, img: np.ndarray, x, y, crop_radius, src_bayer_pattern: BayerPattern = BayerPattern.BGGR) -> np.ndarray:
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

        H0, W0 = img.shape
        rH, rW = crop_radius, crop_radius

        x0, y0 = x, y
        x0, y0 = x0 // 2 * 2 + crop_x_offset, y0 // 2 * 2 + crop_y_offset

        img_crop = img[y0-rH:y0+rH+1, x0-rW:x0+rW+1]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)
            
        if img_crop.ndim == 2:
            img_crop = np.expand_dims(img_crop, axis=-1)

        return img_crop
        
        # Placeholder method for processing each block of image data
        # You can implement your specific processing logic here

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
    
class CleanRawImages(Dataset):

    def __init__(self, *, index_file: Optional[str]=None, data_txt: Optional[SmartPath], opts: DataAugOptions):
        """
        Args:
            - data_dir: a directory that contains "index.json" and raw images
            - index_file: the absolute path to the index file
        """
        super().__init__()

        # assert not (index_file is None and data_dir is None)

        # if data_dir is None:
        #     index_file = SmartPath(index_file)
        # else:
        #     assert index_file is None
        #     index_file = data_dir / "index.json"

        self.opts = DataAugOptions()
        self.filelist: List[RawImageItem] = []
        self.filelist: List = []
        
        with open(data_txt, 'r') as f:
            raw_paths = f.read().splitlines()
        
        for raw_path in raw_paths:
            if not raw_path.endswith('.ARW'):
                continue
            # raw_path = os.path.join(data_dir, raw_name)
            raw_info = rawpy.imread(raw_path)
            # width = raw_info.sizes.raw_width
            # height = raw_info.sizes.raw_height
            height, width = raw_info.raw_image_visible.shape
            black_level = raw_info.black_level_per_channel[0]
            white_level = raw_info.white_level
            bayer_pattern = raw_info.color_desc.decode()
            if bayer_pattern == 'RGBG':
                bayer_pattern = 'RGGB'
            bayer_pattern = BayerPattern(bayer_pattern)
            g_mean_01 = raw_info.raw_image_visible.mean() / white_level
            item = RawImageItem(path=raw_path,
                                width=width,
                                height=height,
                                black_level=black_level,
                                white_level=white_level,
                                bayer_pattern=bayer_pattern,
                                g_mean_01=g_mean_01)            
            self.filelist.append(item)
        print(f'Numboer of RawImages: {len(self.filelist)} have been loaded')
        # with index_file.open() as f:
        #     items = [RawImageItem.parse_obj(x) for x in json.load(f)]
        #     for item in items:
        #         if data_dir is not None:
        #             item.path = str(data_dir / item.path)
        #         self.filelist.append(item)

    def __len__(self):
        return len(self.filelist)
    
    def random_flip_and_crop(self, img: np.ndarray, src_bayer_pattern: BayerPattern) -> np.ndarray:
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

        H0, W0 = img.shape
        tH, tW = self.opts.output_shape

        x0, y0 = np.random.randint(0, W0 - tW), np.random.randint(0, H0 - tH)
        x0, y0 = x0 // 2 * 2 + crop_x_offset, y0 // 2 * 2 + crop_y_offset

        img_crop = img[y0:y0+tH, x0:x0+tW]
        if flip_lr:
            img_crop = np.flip(img_crop, axis=1)
        if flip_ud:
            img_crop = np.flip(img_crop, axis=0)

        return img_crop

    def __getitem__(self, index: int):
        item = self.filelist[index]
        rawimg = rawpy.imread(item.path).raw_image_visible
        # buf = smart_load_from(item.path)
        # rawimg = np.fromfile(buf, dtype=np.uint16).reshape((item.height, item.width))
        
        # random crop to output size
        rawimg = self.random_flip_and_crop(rawimg, item.bayer_pattern).astype(np.float32)

        raw01 = (rawimg - item.black_level) / (item.white_level - item.black_level)
        H, W = raw01.shape
        # pixel shuffle to RGGB image
        rggb01 = raw01.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)
        return rggb01, np.array(item.g_mean_01)


class NoiseProfileFunc:

    def __init__(self, noise_profile: NoiseProfile):
        self.polyK = np.poly1d(noise_profile.K)
        self.polyB = np.poly1d(noise_profile.B)
        self.value_scale = noise_profile.value_scale

    def __call__(self, iso, value_scale=959.0):
        r = value_scale / self.value_scale
        k = self.polyK(iso) * r
        b = self.polyB(iso) * r * r

        return k, b


class DataAug:

    def __init__(self, opts: DataAugOptions):
        self.opts = opts
        self.noise_func = NoiseProfileFunc(opts.noise_profile)

    def transform(self, batch_img01: np.ndarray, batch_g_mean: float, mode='vi') -> Tuple[mge.Tensor, mge.Tensor, mge.Tensor]:
        """
        Args:
            - img: [-black/camera_value_scale, 1.0]

        Returns:
            - noisy_img
            - iso
        """
        
        batch_img01 = np.transpose(batch_img01, (0, 3, 1, 2))
        batch_imgs = mge.tensor(batch_img01) * self.opts.camera_value_scale
        batch_gt = self.brightness_aug(batch_imgs, batch_g_mean)
        # print(f'batch_gt size: {batch_gt.shape}')
        batch_imgs, batch_iso = self.add_noise(batch_gt)
        cvt_k, cvt_b = self.k_sigma(batch_iso)
        # print(f'cvt_k: {cvt_k}, cvt_b: {cvt_b}')
        # print(f'cvt_k.dtype: {cvt_k.dtype}, cvt_b.dtype: {cvt_b.dtype}')
        cvt_k = cvt_k.astype(np.float32)
        cvt_b = cvt_b.astype(np.float32)
        # print(f'new - cvt_k: {cvt_k}, cvt_b: {cvt_b}')

        batch_imgs = batch_imgs * cvt_k.reshape(-1, 1, 1, 1) + cvt_b.reshape(-1, 1, 1, 1)
        batch_gt = batch_gt * cvt_k.reshape(-1, 1, 1, 1) + cvt_b.reshape(-1, 1, 1, 1)
        batch_imgs /= self.opts.camera_value_scale
        batch_gt /= self.opts.camera_value_scale
        if mode == 'vi':
            b, c, h, w = batch_gt.shape
            h_radius, w_radius = (np.array([h, w])-1)//2
            batch_gt = batch_gt[:, :, h_radius, w_radius].reshape(b, c, 1, 1)
        return (batch_imgs, batch_gt, cvt_k)

    def k_sigma(self, iso: float) -> Tuple[float, float]:
        k, sigma = self.noise_func(iso, value_scale=self.opts.camera_value_scale)
        k_a, sigma_a = self.noise_func(self.opts.anchor_iso, value_scale=self.opts.camera_value_scale)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        return cvt_k, cvt_b

    def brightness_aug(self, img_batch: mge.Tensor, orig_gmean: List[float]) -> mge.Tensor:
        low, high = self.opts.target_brighness_range
        N = len(orig_gmean)
        btarget = np.exp(np.random.uniform(np.log(low), np.log(high), size=(N, )))
        s = np.clip(btarget / orig_gmean, 0.01, 1.0, dtype=np.float32)
        # print(f'brightness - orig_gmean: {orig_gmean}, img_batch_dtype: {img_batch.dtype}, s.dtype: {s.dtype}')
        return img_batch * s.reshape(-1, 1, 1, 1)

    def add_noise(self, img: mge.Tensor) -> Tuple[mge.Tensor, float]:
        """
        Args:
            - img: [-black, camera_value_scale]

        Returns:
            - noisy_img
            - iso
        """

        N = img.shape[0]
        isos = np.random.uniform(*self.opts.iso_range, size=(N, ))
        k, b = self.noise_func(isos, value_scale=self.opts.camera_value_scale)
        k = k.reshape(-1, 1, 1, 1).astype(np.float32)
        b = b.reshape(-1, 1, 1, 1).astype(np.float32)

        # print(f'img.dtype: {img.dtype}, k: {k}, b: {b}')
        # print(f'img max: {(img).max()}, img min: {(img).min()}')
        # print(f'img/k max: {(img/k).max()}, img/k min: {(img/k).min()}')
        # shot_noisy = megengine.random.poisson((img / k).clip(0, 1)) * k
        shot_noisy = megengine.random.poisson(img / k) * k
        read_noisy = megengine.random.normal(size=img.shape) * np.sqrt(b)
        noisy = shot_noisy + read_noisy
        noisy = F.round(noisy)

        return noisy, isos
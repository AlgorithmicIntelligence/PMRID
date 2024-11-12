import os
import numpy as np
import cv2

from train_patch import generate_validation_set
from dataset.training import CleanRawImages, DataAug, DataAugOptions
import megengine as mge
# from models.net_mge_org import Network
from models.net_mge_light import Network
from tqdm import tqdm
import pickle
from utils import RawUtils
import rawpy
from megengine.utils.module_stats import module_stats
from src.python.data_processing import DataLoader, DataProcessor
from datetime import datetime
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

t_info = datetime.now()
time_message = str(t_info.year) + \
                str(t_info.month).zfill(2) + \
                str(t_info.day).zfill(2) + \
                str(t_info.hour).zfill(2) + \
                str(t_info.minute).zfill(2) + \
                str(t_info.second).zfill(2)

def generate_testing_set(valid_txt:str, aug_obj:DataAug):
    padding_radius = 6
    white_level = 16383
    black_level = 512
    
    with open(valid_txt, 'r') as f:
        image_infos = [line.split() for line in f.read().splitlines()]    

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
    return imgs, gt, norm_k

def inference_vi(raw_path, model_path='checkpoints/20240418095748/epoch8_iter168000_trainingloss_0.002520_validloss_0.003667.pkl', testing_txt='/home/user/work/data/SID/Sony_test_list_raw.txt'):
    padding_radius = 6
    white_level = 1023
    black_level = 0   
    iso = 6400
    
    aug_opts = DataAugOptions()
    testing_aug = DataAug(aug_opts)
    
    net = Network()
    with open(model_path, 'rb') as f:
        states = pickle.load(f)
    net.load_state_dict(states)
    net.eval()
    input_data = np.random.rand(1, 1, 13, 13).astype(np.float32)
    
    total_stats, stats_details = module_stats(
    net,
    inputs=input_data,
    cal_params=True,
    cal_flops=True,
    cal_activations=True,
    logging_to_stdout=True,
    )
    print("params {} flops {} acts {}".format(total_stats.param_dims, total_stats.flops, total_stats.act_dims))

    output_folder_path = 'results'
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
    output_name = os.path.splitext(os.path.basename(raw_path))[0] + f'_iso{iso}_pred.raw'
    output_basename = os.path.join(output_folder_path, output_name)    
        
        
    rawimg = np.fromfile(raw_path, np.uint16)[:1200*1600].reshape(1200, 1600)
    rawimg = np.pad(rawimg, ((padding_radius, padding_radius), (padding_radius, padding_radius)), mode='reflect')
    rawimg = (rawimg.astype(np.int32) - black_level) / (white_level - black_level)
    
    raw_mean = rawimg.mean()
    rawimg = np.expand_dims(rawimg, axis=(0,-1))
    raw_mean = np.expand_dims(raw_mean, axis=0)
    # print(f'real iso: {iso}')
    imgs, gt, norm_k, norm_b = testing_aug.transform(rawimg, raw_mean, np.array([iso]), mode='general')
    gt = gt[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
    
    patch_radius = 6
    patch_size = patch_radius*2+1
    batch_size = 2**13
    # batch_size = 1
                
    b, c, h, w = gt.shape
    output = list()
    batch_input = list()
    for pixel_index in tqdm(range(h*w), dynamic_ncols=True):
        y = pixel_index // w
        x = pixel_index % w
        
        input = imgs[0, :, y:y+patch_size, x:x+patch_size]
        batch_input.append(input)
        if len(batch_input) == batch_size or pixel_index == h*w-1:
            batch_input = mge.tensor(np.stack(batch_input))
            pred = net(batch_input)
            # pred = (pred - norm_b) / norm_k
            output.extend(pred)
            
            # print(f'pred: {len(output)}, {pred.flatten().shape}, {pred.dtype}, {pred.min()}, {pred.max()}')
            batch_input = list()
                
    pred = np.array(output, dtype=np.float32).reshape(h, w)
    pred = ((pred * 959 - norm_b) / norm_k)/959
    pred =  pred * (white_level - black_level) + black_level
    # pred = pred / 16383        
    pred = pred * 4
    pred = np.clip(np.array(pred), 0, 4095)
    pred = np.round(pred)
    pred = pred.astype(np.uint16)
    with open(output_basename, 'wb') as f:
        f.write(pred.tobytes())    
    
def inference(model_path='checkpoints/20240418095748/epoch8_iter168000_trainingloss_0.002520_validloss_0.003667.pkl', testing_txt='/home/user/work/data/SID/Sony_test_list_raw.txt'):
    aug_opts = DataAugOptions()
    testing_aug = DataAug(aug_opts)
    
    net = Network()
    with open(model_path, 'rb') as f:
        states = pickle.load(f)
    net.load_state_dict(states)
    net.eval()

    output_folder_path = 'results'
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)
        
    padding_radius = 6
    white_level = 16383
    black_level = 512
    
    with open(testing_txt, 'r') as f:
        image_infos = [line.split() for line in f.read().splitlines()]    
        
    
    for image_info in tqdm(image_infos, dynamic_ncols=True):
        image_path, iso = image_info
        iso = int(iso[3:])
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        arw_path = os.path.join('/home/user/work/data/SID/Sony/long', image_name.split('_h')[0]+'.ARW')
        output_basename = os.path.join(output_folder_path, image_name)
        
        # print(arw_path, os.path.isfile(arw_path))
        raw_info = rawpy.imread(arw_path)
        wb_gain = np.array(raw_info.camera_whitebalance)[:3] / 1024
        ccm = raw_info.color_matrix        
        
        print(image_name)
        rawimg = np.fromfile(image_path, np.uint16).reshape(2848, 4256)
        rawimg = np.pad(rawimg, ((padding_radius, padding_radius), (padding_radius, padding_radius)), mode='reflect')
        # print(f'orig - rawMax: {rawimg.max()}, rawMin: {rawimg.min()}')
        rawimg = (rawimg.astype(np.int32) - black_level) / (white_level - black_level)

        raw_mean = rawimg.mean()
        rawimg = np.expand_dims(rawimg, axis=(0,-1))
        raw_mean = np.expand_dims(raw_mean, axis=0)
        # print(f'real iso: {iso}')
        imgs, gt, norm_k, norm_b = testing_aug.transform(rawimg, raw_mean, np.array([iso]), mode='general')
        gt = gt[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        
        patch_radius = 6
        patch_size = patch_radius*2+1
        batch_size = 2**13
        # batch_size = 1
                  
        b, c, h, w = gt.shape
        output = list()
        batch_input = list()
        for pixel_index in tqdm(range(h*w), dynamic_ncols=True):
            y = pixel_index // w
            x = pixel_index % w
            
            input = imgs[0, :, y:y+patch_size, x:x+patch_size]
            batch_input.append(input)
            if len(batch_input) == batch_size or pixel_index == h*w-1:
                batch_input = mge.tensor(np.stack(batch_input))
                pred = net(batch_input)
                # pred = (pred - norm_b) / norm_k
                output.extend(pred)
                
                # print(f'pred: {len(output)}, {pred.flatten().shape}, {pred.dtype}, {pred.min()}, {pred.max()}')
                batch_input = list()
                
                
        pred = np.array(output, dtype=np.float32).reshape(h, w)
        pred = ((pred * 959 - norm_b) / norm_k)/959
        pred =  pred * (white_level - black_level) + black_level
        pred = pred / 16383        
                  
        input = ((imgs * 959 - norm_b) / norm_k)/959
        input =  input * (white_level - black_level) + black_level
        input = input / 16383
        golden = ((gt * 959 - norm_b) / norm_k)/959 
        golden = golden * (white_level - black_level) + black_level
        golden = golden / 16383
        # print(f'outer - inputMax: {input.max().item()}, inputMin: {input.min().item()}, goldenMax: {golden.max().item()}, goldenMin: {golden.min().item()}')
        # imgs, gt, norm_k, norm_b = testing_aug.transform(rawimg, raw_mean, mode='general')
        # golden = golden[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        input = input[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        # print(f'input info: {input.dtype}, {input.shape}, {input.mean()}, {input.max()}, {input.min()}')
        input = np.clip(np.array(input), 0, 1)
        golden = np.clip(np.array(golden), 0, 1)
        pred = np.clip(np.array(pred), 0, 1)
        # print(f'input after info: {input.dtype}, {input.shape}, {golden.shape}, {input.mean()}, {input.max()}, {input.min()}')
        
        # with open(output_basename + '_input.raw', 'wb') as f:
        #     f.write(input.tobytes())
        # with open(output_basename + '_golden.raw', 'wb') as f:
        #     f.write(golden.tobytes())
        input = input[0, 0]
        golden = golden[0, 0]
        
        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
                    input, pred, golden,
                    wb_gain=wb_gain, CCM=np.eye(3),
                )
        print(f'rgb: {wb_gain}, {inp_rgb.shape}, {inp_rgb.dtype}, {inp_rgb.min()}, {inp_rgb.max()}')
        cv2.imwrite(output_basename + '_input.bmp', (inp_rgb*255).astype(np.uint8)[...,::-1])
        cv2.imwrite(output_basename + '_golden.bmp', (gt_rgb*255).astype(np.uint8)[...,::-1])
        cv2.imwrite(output_basename + '_pred.bmp', (pred_rgb*255).astype(np.uint8)[...,::-1])
     
def inference_image(model_path='checkpoints/20240418095748/epoch8_iter168000_trainingloss_0.002520_validloss_0.003667.pkl', testing_txt='/home/user/work/data/SID/Sony_test_list_raw.txt'):
    dataset_id = 'vicore'
    # dataset_id = 'general'
    model_path = 'checkpoints/modelx16/epoch8000_iter162_trainingloss_0.002312_validloss_0.003285.pkl'
    model_path = 'checkpoints/20240813103328/epoch5116_iter162_trainingloss_0.003895_validloss_0.003551.pkl'
    # model_path = 'checkpoints/20240705110154/epoch8000_iter162_trainingloss_0.002312_validloss_0.003285.pkl'
    # model_path = 'checkpoints/20240805172940/epoch441_iter162_trainingloss_0.002504_validloss_0.003389.pkl'
    # model_path = 'checkpoints/20240715164932/epoch100_iter162_trainingloss_0.007024_validloss_0.007264.pkl'
    
    # model_path = 'checkpoints/20240715164932/epoch3_iter162_trainingloss_0.013292_validloss_0.015745.pkl'
    # model_path = 'checkpoints/20240809171358/epoch1414_iter162_trainingloss_0.004409_validloss_0.003814.pkl'
    aug_opts = DataAugOptions()
    testing_aug = DataAug(aug_opts)
    
    noise_k = (0.0005995267, 0.00868861)
    noise_b = (7.11772e-7, 6.514934e-4, 0.11492713)
    aug = DataProcessor(noise_k, noise_b)
    
    net = Network()
    with open(model_path, 'rb') as f:
        states = pickle.load(f)
    net.load_state_dict(states)
    net.eval()

    input_data = np.random.rand(1, 4, 1088, 1920).astype(np.float32)
    
    total_stats, stats_details = module_stats(
    net,
    inputs=input_data,
    cal_params=True,
    cal_flops=True,
    cal_activations=True,
    logging_to_stdout=True,
    )
    print("params {} flops {} acts {}".format(total_stats.param_dims, total_stats.flops, total_stats.act_dims))

    output_folder_path = os.path.join('results', time_message)
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
            

        
    if dataset_id == 'vicore':
        # load vicore dataset - start
        white_level = 1023
        # white_level = 4095
        black_level = 0   
        iso = 8000
        raw_path = '/home/user/work/data/3140/capture/20240625095017/sensor#0/stream#0/raw/RG10_frameidx13824_time556475557_w1600_h1200_gain16335_shutter16667_ledstatus0_ledidx0_projectoridx-1_gainr416_gaing256_gainb485_ct5789_bv487_hdr0_gainshort0_shuttershort0.raw'
        # raw_path = '/home/user/work/data/Tsing/3-51200/linear/raws/RG12_frameidx1_w1920_h1080_gain2244_gainr400_gaing256_gainb470_ct6500_bv700_vicore.raw'
        # txt_path = '/home/user/work/github/PMRID/dataset_iso51200.txt'
        # image_infos = list()
        # with open(txt_path, 'r') as f:
        #     for line in f:
        #         image_infos.append([line.strip(), f'ISO{iso}'])                  
        image_infos = [[raw_path, f'ISO{iso}']]
        
        # load vicore dataset - end
    else:
        white_level = 16383
        black_level = 512            
        with open(testing_txt, 'r') as f:
            image_infos = [line.split() for line in f.read().splitlines()]    
    
    
    for image_info in tqdm(image_infos, dynamic_ncols=True):
        image_path, iso = image_info
        iso = int(iso[3:])
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_basename = os.path.join(output_folder_path, image_name)
        
        if dataset_id == 'vicore':
            h, w = 1200, 1600
            # h, w = 1080, 1920
            p_h = (np.ceil(h/32) * 32 - h) // 2
            p_w = (np.ceil(w/32) * 32 - w) // 2
            p_h, p_w = int(p_h), int(p_w)
            
            rawimg = np.fromfile(image_path, np.uint16)[:h*w].reshape(h, w)
            rawimg = np.pad(rawimg, ((p_h, p_h), (p_w, p_w)), mode='reflect')
            h, w = rawimg.shape
            # rawimg = (rawimg.astype(np.int32) - black_level) / (white_level - black_level)
            wb_gain = np.array([416, 256, 485]) / 256   
            ccm =  np.array([[265,34,-44],[16,221,19],[-45,-16,317]])/256
            ccm = np.eye(3)
        else:          
            arw_path = os.path.join('/home/user/work/data/SID/Sony/long', image_name.split('_h')[0]+'.ARW')
            raw_info = rawpy.imread(arw_path)
            wb_gain = np.array(raw_info.camera_whitebalance)[:3] / 1024
            ccm = raw_info.color_matrix  
            ccm = np.eye(3)
            h = 2848
            w = 4256
            rawimg = np.fromfile(image_path, np.uint16).reshape(h, w)
        # rawimg = np.pad(rawimg, ((padding_radius, padding_radius), (padding_radius, padding_radius)), mode='reflect')
        # print(f'orig - rawMax: {rawimg.max()}, rawMin: {rawimg.min()}')
        rawimg = (rawimg.astype(np.int32) - black_level) / (white_level - black_level)
        rggb = rawimg.reshape(-1, h//2, 2, w//2, 2).transpose(0, 1, 3, 2, 4).reshape(-1, h//2, w//2, 4)
        rggb_mean = rggb.mean(axis=(1, 2))
        images_g_mean = rggb_mean[:, 1:3].mean(axis=1)

        # raw_mean = rawimg.mean()
        # rawimg = np.expand_dims(rawimg, axis=(0,-1))
        # raw_mean = np.expand_dims(raw_mean, axis=0)
        # print(f'real iso: {iso}')
        imgs, gt, norm_k, norm_b = aug.transform(rggb, images_g_mean, isos=[iso], mode='test')

        pred = net(imgs)

        # pred = np.array(pred, dtype=np.float32).reshape(h, w)
        pred = ((pred * 959 - norm_b) / norm_k)/959
        pred =  pred * (white_level - black_level) + black_level
        pred_bayer = pred
        pred = pred / white_level  
        
        pred_bayer = np.clip(np.array(pred_bayer[0]), 0, white_level)
        pred_bayer = np.round(pred_bayer/white_level*1023)
        pred_bayer = pred_bayer.astype(np.uint16)
        pred_bayer = pred_bayer.transpose(1, 2, 0)
        pred_bayer = pred_bayer.reshape(h//2, w//2, 2, 2).transpose(0, 2, 1, 3).reshape(h, w)
        pred_bayer = pred_bayer[p_h:h-p_h, p_w:w-p_w]

        with open(output_basename +  f'_iso{iso}pred.raw', 'wb') as f:
            f.write(pred_bayer.tobytes())        
        
        input = ((imgs * 959 - norm_b) / norm_k)/959
        input =  input * (white_level - black_level) + black_level
        input = input / white_level
        
        golden = ((gt * 959 - norm_b) / norm_k)/959 
        golden = golden * (white_level - black_level) + black_level
        golden = golden / white_level
        # print(f'outer - inputMax: {input.max().item()}, inputMin: {input.min().item()}, goldenMax: {golden.max().item()}, goldenMin: {golden.min().item()}')
        # imgs, gt, norm_k, norm_b = testing_aug.transform(rawimg, raw_mean, mode='general')
        # golden = golden[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        # input = input[:, :, padding_radius:-padding_radius, padding_radius:-padding_radius]
        # print(f'input info: {input.dtype}, {input.shape}, {input.mean()}, {input.max()}, {input.min()}')
        input = np.clip(np.array(input), 0, 1)
        golden = np.clip(np.array(golden), 0, 1)
        pred = np.clip(np.array(pred), 0, 1)
        # print(f'input after info: {input.dtype}, {input.shape}, {golden.shape}, {input.mean()}, {input.max()}, {input.min()}')
        
        # with open(output_basename + '_input.raw', 'wb') as f:
        #     f.write(input.tobytes())
        # with open(output_basename + '_golden.raw', 'wb') as f:
        #     f.write(golden.tobytes())
        input = input[0]
        golden = golden[0]
        pred = pred[0]
        
        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
                    input, pred, golden,
                    wb_gain=wb_gain, CCM=ccm,
                )
        # raw_image_rgb = raw_info.postprocess(output_bps=16, no_auto_bright=True)
        # print(f'raw rgb: {raw_image_rgb.shape}')
        # print(f'rgb: {wb_gain}, {inp_rgb.shape}, {inp_rgb.dtype}, {inp_rgb.min()}, {inp_rgb.max()}')
        cv2.imwrite(output_basename + f'_iso{iso}_input.bmp', (inp_rgb*255).astype(np.uint8)[...,::-1])
        cv2.imwrite(output_basename + f'_iso{iso}_golden.bmp', (gt_rgb*255).astype(np.uint8)[...,::-1])
        cv2.imwrite(output_basename + f'_iso{iso}_pred.bmp', (pred_rgb*255).astype(np.uint8)[...,::-1])
                
if __name__ == '__main__':
    # inference()
    # raw_path = '/home/user/work/data/3140/capture/20240625095017/sensor#0/stream#0/raw/RG10_frameidx13824_time556475557_w1600_h1200_gain16335_shutter16667_ledstatus0_ledidx0_projectoridx-1_gainr416_gaing256_gainb485_ct5789_bv487_hdr0_gainshort0_shuttershort0.raw'
    # inference_vi(raw_path)
    inference_image()

from torch.utils.data import Dataset

import numpy as np
import torch 
import os
import rasterio
import torch.nn.functional as F
from torchvision import transforms

mean = [1315.1941, 1320.8625, 1322.4388, 1312.7904, 1321.5713, 1331.3823, 1326.7014, 1314.1525, 1315.7151, 1313.9711]
std = [912.0858, 915.2389, 918.5796, 915.2799, 919.2444, 922.5997, 921.9182, 920.0427, 917.4285, 913.5229]
def pad_to_stride(tensor, stride):
    _, h, w = tensor.shape
    pad_h = (stride - h % stride) 
    pad_w = (stride - w % stride)
    padding = (0, pad_w, 0, pad_h)  # (w_left, w_right, h_top, h_bottom)
    return torch.nn.functional.pad(tensor, padding,mode='reflect')
def prepare(path_images,path_masks,files):
    output_stride = 256
    result=None
    result_m=None
    
    
    for i in files:
        with rasterio.open(f'{path_images}/{i}') as src:
            image = src.read().astype(np.float32)  # Read image
            image = torch.tensor(image)
    
            image = pad_to_stride(image, output_stride)
            h_splits = image.shape[1] // output_stride
            w_splits = image.shape[2] // output_stride
            tensor_split = image.unfold(1, output_stride, output_stride).unfold(2, output_stride, output_stride)
            tensor_split = tensor_split.contiguous().view(-1, image.shape[0], output_stride, output_stride)
        with rasterio.open(f'{path_masks}/{i}') as src:
            image_m = src.read().astype(np.float32)  # Read image
            image_m = torch.tensor(image_m)
    
            image_m = pad_to_stride(image_m, output_stride)
            h_splits_m = image_m.shape[1] // output_stride
            w_splits_m = image_m.shape[2] // output_stride
            tensor_split_m = image_m.unfold(1, output_stride, output_stride).unfold(2, output_stride, output_stride)
            tensor_split_m = tensor_split_m.contiguous().view(-1, image_m.shape[0], output_stride, output_stride)
    
        # Model evaluation
        if result is not None:
            result=torch.cat((result, tensor_split), dim=0)
            result_m=torch.cat((result_m, tensor_split_m), dim=0)
        else:
            result=tensor_split
            result_m=tensor_split_m
    return result,result_m
class WaterDataset(Dataset):
    def __init__(self, img_path, mask_path, file_names):
        self.result, self.result_m  = prepare(img_path,mask_path,file_names)
        self.trans = transforms.Normalize(mean=mean, std=std)
    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):

        
        return self.trans(self.result[idx]), self.result_m[idx]




import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils import data
from PIL import Image
import numpy as np
import os
from collections import OrderedDict
import util.util as util
import cv2

from PIL import Image
from base_dataset import BaseDataset, get_params, get_transform
import json

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, stop=10000):
    images = []
    count = 0
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                count += 1
            if count >= stop:
                return images
    return images

class UnpairedDataset(data.Dataset):
    def __init__(self, root, root2, opt, transforms_r=None, mode='train'):

        self.root = root
        self.mode = mode

        all_img = make_dataset(self.root)

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)

        self.opt = opt
        
        if mode == 'train':
            
            self.img2 = make_dataset(root2)

            if len(self.data) > len(self.img2):
                howmanyrepeat = (len(self.data) // len(self.img2)) + 1
                self.img2 = self.img2 * howmanyrepeat
            elif len(self.img2) > len(self.data):
                howmanyrepeat = (len(self.img2) // len(self.data)) + 1
                self.data = self.data * howmanyrepeat
            

            cutoff = min(len(self.data), len(self.img2))

            self.data = self.data[:cutoff] 
            self.img2 = self.img2[:cutoff] 

            self.min_length =cutoff
        else:
            self.min_length = len(self.data)


    def __getitem__(self, index):

        img_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split('.')[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params(self.opt, img_r.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1), norm=False)        

        if self.mode != 'train':
            A_transform = self.transform_r

        img_r = A_transform(img_r )

        B_mode = 'L'
        if self.opt.output_nc == 3:
            B_mode = 'RGB'


        img_normals = 0
        label = 0

        input_dict = {'r': img_r, 'path': img_path, 'index': index, 'name' : base, 'label': label}

        if self.mode=='train':
            cur_path = self.img2[index]
            cur_img = B_transform(Image.open(cur_path).convert(B_mode))
            input_dict['line'] = cur_img

        return input_dict

    def __len__(self):
        return self.min_length



class PairedDataset(data.Dataset):
    def __init__(self, root, root2, opt, transforms_r=None, mode='train'):

        self.root = root
        self.mode = mode

        all_img = make_dataset(self.root)

        self.data = all_img
        self.mode = mode

        self.transform_r = transforms.Compose(transforms_r)

        self.opt = opt
        
        if mode == 'train':
            
            self.img2 = make_dataset(root2)
            
            # rootとroot2の画像のペアを作成する処理を追加
            self.data = self.create_pairs(self.data, self.img2)

            self.min_length = len(self.data)
        else:
            self.min_length = len(self.data)


    def create_pairs(self, list1, list2):
        pairs = []
        for img_path in list1:
            basename = os.path.basename(img_path)
            base = basename.split('.')[0]
            matching_paths = [path for path in list2 if os.path.basename(path).split('.')[0] == base]
            if matching_paths:
                pairs.append((img_path, matching_paths[0]))
        return pairs

    def __getitem__(self, index):

        img_path, img2_path = self.data[index]

        basename = os.path.basename(img_path)
        base = basename.split('.')[0]

        img_r = Image.open(img_path).convert('RGB')
        transform_params = get_params(self.opt, img_r.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.input_nc == 1), norm=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.opt.output_nc == 1), norm=False)        

        if self.mode != 'train':
            A_transform = self.transform_r

        img_r = A_transform(img_r)

        B_mode = 'L'
        if self.opt.output_nc == 3:
            B_mode = 'RGB'

        img2 = B_transform(Image.open(img2_path).convert(B_mode))

        label = 0

        input_dict = {'r': img_r, 'line': img2, 'path': img_path, 'index': index, 'name': base, 'label': label}

        return input_dict

    def __len__(self):
        return self.min_length
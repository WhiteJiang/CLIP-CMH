# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger

import numpy
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import h5py
import clip

logger = getLogger()


class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)


def text_transform(text):
    return text


class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
            self,
            path,
            return_index=False,
            partition='train'
    ):
        self.path = path
        self.partition = partition
        training = 'train' in partition.lower()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if training:
            trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            trans = transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        self.trans = trans
        self.return_index = return_index
        image = open(os.path.join(self.path, "nus_top10_images.txt"), "r")
        self.imgs = image.readlines()
        labels = open(os.path.join(self.path, "nus_top10_labels.txt"), "r")
        self.labels = labels.readlines()
        texts = open(os.path.join(self.path, "nus_top10_tags.txt"), "r")
        self.texts = texts.readlines()
        if 'train' in partition.lower():
            self.imgs = self.imgs[:-2000]
            self.texts = self.texts[:-2000]
            self.labels = self.labels[:-2000]
            self.length = len(self.labels) - 2000
        else:
            self.imgs = self.imgs[-2000:]
            self.texts = self.texts[-2000:]
            self.labels = self.labels[-2000:]
            self.length = 2000
        self.text_dim = 1024

    def __getitem__(self, index):
        image = self.imgs[index]
        image = Image.open(os.path.join(self.path, image)).convert('RGB')
        text = self.texts[index]
        text = text.replace('\n', '').replace('\r', '')
        text = text.split(" ")

        multi_crops = self.trans(image)

        label = self.labels[index]
        label = label.replace('\n', '').replace('\r', '')
        label = label.split(" ")
        label = [int(i) for i in label]
        label = numpy.array(label)

        if self.return_index:
            return index, multi_crops, text, label
        return multi_crops, text, label
        # return multi_crops, text, index

    def __len__(self):
        return self.length


def MIRFlickr25K_fea(partition):
    root = 'D:/迅雷下载/MIRFLICKR25K/'
    data_img = sio.loadmat(os.path.join(root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(root, 'mirflickr25k-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(root, 'mirflickr25k-lall.mat'))['LAll']

    test_size = 2000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels


def NUSWIDE_fea(partition):
    root = 'D:/迅雷下载/NUS-WIDE-TC10/'
    test_size = 2100
    data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll']
    data_txt = sio.loadmat(root + 'nus-wide-tc10-yall.mat')['YAll']
    labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']

    test_size = 2100
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size::], data_txt[-test_size::], labels[-test_size::]
    else:
        data_img, data_txt, labels = data_img[0: -test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels


if __name__ == '__main__':
    image = open("nus_top10_tags.txt", "r")
    lines = image.readlines()
    for line in lines:
        line = line.replace('\n', '').replace('\r', '')
        clip.tokenize(line)
    # text = lines[1].replace('\n', '').replace('\r', '')
    # text = text.split(" ")
    # text = [int(i) for i in text]
    # print(text)
    # text = numpy.array(text)
    # print(text)

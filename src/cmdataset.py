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

def text_transform(text):
    return text


class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
            self,
            path='../ImageData/Flickr/',
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
            # print(len(self.labels), len(self.texts), len(self.imgs))
        else:
            self.imgs = self.imgs[-2000:]
            self.texts = self.texts[-2000:]
            self.labels = self.labels[-2000:]
            # print(len(self.labels), len(self.texts), len(self.imgs))
            self.length = 2000
        self.text_dim = 1024

    def __getitem__(self, index):
        image = self.imgs[index]
        image = image.replace('\n', '').replace('\r', '')
        image = Image.open(os.path.join(self.path, image)).convert('RGB')
        text = self.texts[index]
        text = text.replace('\n', '').replace('\r', '')
        text = clip.tokenize(text)

        multi_crops = self.trans(image)

        label = self.labels[index]
        label = label.replace('\n', '').replace('\r', '')
        label = label.split(" ")
        label = [int(i) for i in label]
        label = numpy.array(label)

        return {
            "image": multi_crops,
            "text": text,
            "label": label,
            "index": index,
        }

    def __len__(self):
        return self.length


if __name__ == '__main__':
    image = open("nus_top10_labels.txt", "r")
    lines = image.readlines()
    text = lines[1].replace('\n', '').replace('\r', '')
    text = text.split(" ")
    text = [int(i) for i in text]
    print(text)
    text = numpy.array(text)
    print(text)

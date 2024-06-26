from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import cv2
import numpy as np

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, stargan_selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.stargan_selected_attrs = stargan_selected_attrs
        self.mode = mode
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        self.train_end = 1000
        self.val_end = 1200
        self.test_end = 1300
        # self.train_end = 10
        # self.val_end = 20
        # celebamask-HQ只有30000张，取前面2000张作为测试集
        if self.mode == 'train':
            # self.images = images[:182637]
            # self.labels = labels[:182637]
            self.images = images[:self.train_end]
            self.labels = labels[:self.train_end]
            # self.images = images[:28000]
            # self.labels = labels[:28000]
        if self.mode == 'val':
            self.images = images[self.train_end:self.val_end]
            self.labels = labels[self.train_end:self.val_end]
        if self.mode == 'test':
            # self.images = images[182637:]
            # self.labels = labels[182637:]
            self.images = images[self.val_end:self.test_end]
            self.labels = labels[self.val_end:self.test_end]
            # self.images = images[28000:]
            # self.labels = labels[28000:]
        
        # self.tf = transforms.Compose([
        #     # transforms.CenterCrop(170), #变成celebamask-HQ所以不粗要crop
        #     transforms.Resize(image_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])

        # stargan
        self.attr2idx = {}
        self.idx2attr = {}
        self.test_dataset = []
        self.train_dataset = []
        self.val_dataset = []
        self.preprocess()

        if self.mode == 'train':
            self.num_images = len(self.train_dataset)
        elif self.mode == 'val':
            self.num_images = len(self.val_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines_all = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines_all[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        # print(lines[1])
        # lines = lines[182637:]
        # lines = lines[28002:]
        # lines = lines[182639:]

        lines = lines_all[2:]
        # random.seed(1234)
        # random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            # print(filename)
            # print(filename)
            values = split[1:]

            label = []
            for attr_name in self.stargan_selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            if i+1 <= self.train_end:
                self.train_dataset.append([filename, label])
            elif i+1 <= self.val_end:
                self.val_dataset.append([filename, label])
            elif i+1 <= self.test_end:
                self.test_dataset.append([filename, label])


        
        # print(len(self.test_dataset))
        # print(self.test_dataset[0][0])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        tf = []
        if self.mode == 'train':
            tf.append(transforms.RandomHorizontalFlip(0.5))
        # 测试集不用水平翻转
        # transform.append(transforms.CenterCrop(crop_size))
        tf.append(transforms.Resize(256))
        tf.append(transforms.ToTensor())
        tf.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(tf)
        if self.mode == 'test':
            img = transform(Image.open(os.path.join(self.data_path, self.images[index])))
            att = torch.tensor((self.labels[index] + 1) // 2)
            filename, label = self.test_dataset[index]
        if self.mode == 'val':
            img = transform(Image.open(os.path.join(self.data_path, self.images[index])))
            att = torch.tensor((self.labels[index] + 1) // 2)
            filename, label = self.val_dataset[index]
        if self.mode == 'train':
            img = transform(Image.open(os.path.join(self.data_path, self.images[index])))
            att = torch.tensor((self.labels[index] + 1) // 2)
            filename, label = self.train_dataset[index]

        return img, att, torch.FloatTensor(label),filename
        
    def __len__(self):
        return self.num_images




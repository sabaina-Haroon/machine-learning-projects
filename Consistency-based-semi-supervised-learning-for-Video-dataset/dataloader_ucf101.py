import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import torchvision.models as models
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import save_image
import pickle
import cv2

import os


class UCF101DataLoader(Dataset):

    def __init__(self, ann_file, transform=None):
        super(UCF101DataLoader, self).__init__()

        self.ann_file = ann_file
        self.train_list = np.load('train.list.npy')


    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        # get random video name

        vid_name = self.train_list[idx]  # list(ann_file.keys())[idx]
        semi = 0
        frame_rate = 30

        if vid_name in self.ann_file:
            # get vid len
            vid_len = len(self.ann_file[vid_name]['annotations'][0]['boxes'])
            sf = self.ann_file[vid_name]['annotations'][0]['sf']
            ef = self.ann_file[vid_name]['annotations'][0]['ef']

            # get frames num
            if vid_len > frame_rate:
                frame_indices = list(range(sf, ef - 3, int(vid_len / frame_rate)))
                annot_indices = list(range(0, vid_len - 3, int(vid_len / frame_rate)))
            else:
                frame_indices = list(range(sf, ef - 3, 1))
                annot_indices = list(range(0, vid_len - 3, 1))
            size_ = len(frame_indices)
            if size_ < frame_rate:
                for i in range(size_, frame_rate):
                    frame_indices.append(frame_indices[len(frame_indices) - 1])
                    annot_indices.append(annot_indices[len(annot_indices) - 1])

            # read video as frames
            cap = cv2.VideoCapture(r'/home/cap6412.student5/UCF-101/train/' + vid_name + '.avi')
            # cap.set(cv2.CAP_PROP_FPS, 5)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            buf = np.empty((len(frame_indices), 300, 300, 3), np.dtype('uint8'))
            target = np.empty((len(annot_indices), 5), np.dtype('float64'))

            fc = 0
            ret = True

            for i in range(0, len(frame_indices)):
                frame_no = frame_indices[i]
                annot_no = annot_indices[i]
                cap.set(1, frame_no)
                ret, img = cap.read()
                dim = (300,300)
                buf[i] = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                target[i, :4] = self.ann_file[vid_name]['annotations'][0]['boxes'][annot_no]
                target[i] = np.array(target[i]).astype('int')
                # since 3rd and 4th values of bbx are offsets
                target[i,2] += target[i,0]
                target[i,3] += target[i,1]
                target[i, 0] /= 320.0
                target[i, 1] /= 240.0
                target[i,2] /= 320.0
                target[i,3] /= 240.0
                target[i, 4] = self.ann_file[vid_name]['annotations'][0]['label']



            cap.release()

            frames = torch.from_numpy((np.divide(buf[0:frame_rate], 255.0)).astype(np.float64)).permute(3, 0, 1, 2)
            frames = frames.type(torch.FloatTensor)

            #JUST pick ground truth for centre frame since we will calculate loss only for that
            centre_idx = int(frame_rate/2)
            targets = torch.from_numpy(target[centre_idx]).type(torch.FloatTensor)

            return frames, targets, semi

        else:
            cap = cv2.VideoCapture(r'/home/cap6412.student5/UCF-101/train/' + vid_name + '.avi')
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if frame_count > frame_rate:
                frame_indices = list(range(0, frame_count - 1, int(frame_count / frame_rate)))
            else:
                frame_indices = list(range(0, frame_count, 1))
            size_ = len(frame_indices)
            if size_ < frame_rate:
                for i in range(size_, frame_rate):
                    frame_indices.append(frame_count - 1)

            buf = np.empty((len(frame_indices), 300, 300, 3), np.dtype('float64'))
            target = np.empty((len(frame_indices), 5), np.dtype('float64'))
            semi = 1

            ret = True

            for i in range(0, len(frame_indices)):
                frame_no = frame_indices[i]
                cap.set(1, frame_no)
                ret, img = cap.read()
                dim = (300,300)
                buf[i] = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cap.release()

            frames = torch.from_numpy((np.divide(buf[0:frame_rate], 255.0)).astype(np.float64)).permute(3, 0, 1, 2)
            frames = frames.type(torch.FloatTensor)

            # JUST pick ground truth for centre frame since we will calculate loss only for that
            centre_idx = int(frame_rate / 2)
            targets = torch.from_numpy(target[centre_idx]).type(torch.FloatTensor)

            return frames, targets, semi




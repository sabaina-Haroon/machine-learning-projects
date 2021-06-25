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


class UCF101DetectorData(Dataset):

    def __init__(self, ann_file, transform=None):
        super(UCF101DetectorData, self).__init__()

        self.ann_file = ann_file
        self.test_list = np.load('test.list.npy')

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, idx):
        vid_name = self.test_list[idx]  # list(ann_file.keys())[idx]
        frame_rate = 30

        # read annotation data against video
        if vid_name in self.ann_file:
            # get vid len
            vid_len = len(self.ann_file[vid_name]['annotations'][0]['boxes'])

            # start frame , end frame
            sf = self.ann_file[vid_name]['annotations'][0]['sf']
            ef = self.ann_file[vid_name]['annotations'][0]['ef']

            # getting specified number of frames between start and end frame
            if vid_len > frame_rate:
                frame_indices = list(range(sf, ef - 3, int(vid_len / frame_rate)))
                annot_indices = list(range(0, vid_len - 3, int(vid_len / frame_rate)))
            else:
                frame_indices = list(range(sf, ef - 3, 1))
                annot_indices = list(range(0, vid_len - 3, 1))

            # if less than specified frame copy end frame
            size_ = len(frame_indices)
            if size_ < frame_rate:
                for i in range(size_, frame_rate):
                    frame_indices.append(frame_indices[len(frame_indices) - 1])
                    annot_indices.append(annot_indices[len(annot_indices) - 1])

            # read video as frames
            cap = cv2.VideoCapture(r'UCF-101/' + vid_name + '.avi')

            buf = np.empty((len(frame_indices), 300, 300, 3), np.dtype('uint8'))
            target = np.empty((len(annot_indices), 5), np.dtype('float64'))

            for i in range(0, len(frame_indices)):
                frame_no = frame_indices[i]
                annot_no = annot_indices[i]
                cap.set(1, frame_no)
                ret, img = cap.read()
                dim = (300, 300)
                buf[i] = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                target[i, 1:5] = self.ann_file[vid_name]['annotations'][0]['boxes'][annot_no]

                # since 3rd and 4rth values of bbx are offsets
                target[i, 3] += target[i, 1]
                target[i, 4] += target[i, 2]
                target[i, 0] = self.ann_file[vid_name]['annotations'][0]['label']
            cap.release()

            frames = torch.from_numpy((np.divide(buf[0:frame_rate], 255.0)).astype(np.float64)).permute(3, 0, 1, 2)
            frames = frames.type(torch.FloatTensor)

            # JUST pick ground truth for centre frame since we will calculate loss only for that
            centre_idx = int(frame_rate / 2)

            return frames, target[centre_idx], vid_name

    # check if video is labeled or not
    def if_video_in_annot(self, idx):
        vid_name = self.test_list[idx]
        if vid_name in self.ann_file:
            return True

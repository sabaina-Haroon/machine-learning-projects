import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import UCF101
import os
from i3dpt import I3D
import warnings
import math
import numpy as np
import cv2


class SSD_CON(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_CON, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = UCF101
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.i3d = base
        # Layer learns to scale the l2 normalized features from 7th layer of i3d
        #self.L2Norm = L2Norm(512, 20)
        self.L2Norm = L2Norm(480, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if phase == 'test':
            # self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            #self.detect = Detect(num_classes, 0, 200, 0.001, 0.15)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x_flip = x.clone().detach()
        x_flip = flip(x_flip,4)

        sources = list()
        loc = list()
        conf = list()

        i3d_map1, i3d_map2 = self.i3d(x)

        i3d_map1 = i3d_map1.mean(2)
        i3d_map2 = i3d_map2.mean(2)

        # taken from 7th layer of i3d
        s = self.L2Norm(i3d_map1)
        sources.append(s)

        #taken from 16th layer of i3d
        sources.append(i3d_map2)

        x = i3d_map2.detach()

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x.detach()).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x.detach()).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # zero_mask = torch.cat([o.view(o.size(0), -1) for o in zero_mask], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        loc = loc.view(loc.size(0), -1, 4)
        conf = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        # basic

        sources_flip = list()
        loc_flip = list()
        conf_flip = list()

        # apply vgg up to conv4_3 relu
        i3d_map1_flip, i3d_map2_flip = self.i3d(x_flip)

        i3d_map1_flip = i3d_map1_flip.mean(2)
        i3d_map2_flip = i3d_map2_flip.mean(2)

        # taken from 7th layer of i3d
        s_flip = self.L2Norm(i3d_map1_flip)
        sources_flip.append(s_flip)

        # taken from 16th layer of i3d
        sources_flip.append(i3d_map2_flip)

        x_flip = i3d_map2_flip.detach()

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_flip = F.relu(v(x_flip), inplace=True)
            if k % 2 == 1:
                sources_flip.append(x_flip)

        # apply multibox head to source layers
        for (x_flip, l, c) in zip(sources_flip, self.loc, self.conf):
            append_loc = l(x_flip.detach()).permute(0, 2, 3, 1).contiguous()
            append_conf = c(x_flip.detach()).permute(0, 2, 3, 1).contiguous()
            append_loc = flip(append_loc,2)
            append_conf = flip(append_conf,2)
            loc_flip.append(append_loc)
            conf_flip.append(append_conf)

        loc_flip = torch.cat([o.view(o.size(0), -1) for o in loc_flip], 1)
        conf_flip = torch.cat([o.view(o.size(0), -1) for o in conf_flip], 1)

        loc_flip = loc_flip.view(loc_flip.size(0), -1, 4)

        conf_flip = self.softmax(conf_flip.view(conf.size(0), -1, self.num_classes))


        if self.phase == "test":
            return output
        else:
            return output, conf, conf_flip, loc, loc_flip

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v=='K':
                layers += [nn.Conv2d(in_channels, 256,
                           kernel_size=4, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers



def multibox(extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []

    # adding these two layers for i3d net
    loc_layers += [nn.Conv2d(480, cfg[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(480, cfg[0] * num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(1024, cfg[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, cfg[1] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K'],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_size=(3, 300, 300), mean=0, std=0.15):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size, ) + input_size
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.mean = mean
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(self.mean, std=self.std)
        if x.size(0) == self.noise.size(0):
            return x + self.noise
        else:
            #print('---- Noise Size ')
            return x + self.noise[:x.size(0)]


def build_ssd_con(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_ = I3D(num_classes=400, modality='rgb')
    extras_, head_ = multibox(add_extras(extras[str(size)], 1024), mbox[str(size)], num_classes)
    return SSD_CON(phase, size, base_, extras_, head_, num_classes)


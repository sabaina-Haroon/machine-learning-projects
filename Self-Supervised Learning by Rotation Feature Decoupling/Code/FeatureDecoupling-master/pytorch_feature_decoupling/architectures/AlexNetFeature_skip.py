import pdb

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()


    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class AlexNetFeature(nn.Module):
    def __init__(self, opt):
        super(AlexNetFeature, self).__init__()
        self.skip = nn.Identity()

###### configuration for Experiment Skip connection variant by convolution

        # conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv2 = nn.Sequential(
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        # )
        # pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv3 = nn.Sequential(
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        # )
        # conv4 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # conv5 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv1_up = nn.Sequential(
            nn.Conv2d(64, 384, kernel_size=3, stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2_up = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        #num_pool5_feats = 6 * 6 * 256
        num_pool5_feats = 3 * 3 * 256
        fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(num_pool5_feats, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )

        self._feature_blocks = nn.ModuleList([
            conv1,
            pool1,
            conv1_up,
            conv2,
            pool2,
            conv2_up,
            conv3,
            conv4,
            conv5,
            pool5,
            fc_block,
        ])
        self.all_feat_names = [
            'conv1',
            'pool1',
            'conv1_up',
            'conv2',
            'pool2',
            'conv2_up',
            'conv3',
            'conv4',
            'conv5',
            'pool5',
            'fc_block',
        ]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
    	  x: input image.
    	  out_feat_keys: a list/tuple with the feature names of the features that the function should return. By default the last feature of the network is returned.

    	Return:
            out_feats: If multiple output features were asked then `out_feats` is a list with the asked output features placed in the same order as in `out_feat_keys`. If a single output feature was asked then `out_feats` is that output feature (and not a list).
    	"""
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        feat = self._feature_blocks[0](feat)  # cov1, 64x31x31
        key = self.all_feat_names[0]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[1](feat)  # pool1, 64x15x15
        key = self.all_feat_names[1]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat_conv1up = self._feature_blocks[2](feat)  # con1up, 384x7x7
        key = self.all_feat_names[2]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat_conv1up

        feat = self._feature_blocks[3](feat)  # conv2, 192x15x15
        key = self.all_feat_names[3]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[4](feat)  # pool2, 192x7x7
        key = self.all_feat_names[4]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat_conv2up = self._feature_blocks[5](feat)  # con2up, 256x7x7
        key = self.all_feat_names[5]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat_conv2up

        feat = self._feature_blocks[6](feat)  # conv3, 384x7x7
        key = self.all_feat_names[6]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[7](feat + feat_conv1up)  # conv4, 256x7x7
        key = self.all_feat_names[7]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[8](feat + feat_conv2up)  # conv5, 256x7x7
        key = self.all_feat_names[8]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[9](feat)  # pool5, 256x3x3
        key = self.all_feat_names[9]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[10](feat)  # fcblock, --
        key = self.all_feat_names[10]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat


        # for f in range(max_out_feat+1):
        #     feat = self._feature_blocks[f](feat)
        #     key = self.all_feat_names[f]
        #     if key in out_feat_keys:
        #         out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        return out_feats

    def get_L1filters(self):
        convlayer = self._feature_blocks[0][0]
        batchnorm = self._feature_blocks[0][1]
        filters = convlayer.weight.data
        scalars = (batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05))
        filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()

        return filters

def create_model(opt):
    return AlexNetFeature(opt)

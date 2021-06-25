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

        # conv1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv1_down = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv2 = nn.Sequential(
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        # )
        # pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # conv3 = nn.Sequential(
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(384),
        #     #nn.ReLU(inplace=True),
        # )
        # conv1_3 = nn.Sequential(
        #     #nn.Conv2d(384, (384+64), kernel_size=3, padding=1),
        #     #nn.BatchNorm2d((384+64)),
        #     nn.ReLU(inplace=True),
        # )
        # conv4_prev = nn.Sequential(
        #     nn.Conv2d((384+64), 384, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True),
        # )
        # conv4 = nn.Sequential(
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     #nn.ReLU(inplace=True),
        # )
        # conv2_4 = nn.Sequential(
        #     #nn.Conv2d(256, (256+192), kernel_size=3, padding=1),
        #     #nn.BatchNorm2d((256+192)),
        #     nn.ReLU(inplace=True),
        # )
        # conv5_prev = nn.Sequential(
        #     nn.Conv2d((256 + 192),256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d((256)),
        #     nn.ReLU(inplace=True),
        # )
        # conv5 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        # pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

#################  exp 6 ######################
        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # residual Block #############
        res1_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        res2_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        res3_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64)
        )

        #######   conv1 + res3_1

        res1 = nn.Sequential(
            nn.ReLU(inplace=True),
        )

        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        # residual Block #############
        res1_2 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        res2_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        res3_2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384)
        )

        #######   conv3+ res2
        res2 = nn.Sequential(
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
            nn.Linear(num_pool5_feats, 8024, bias=False),
            nn.BatchNorm1d(8024),
            nn.ReLU(inplace=True),
            nn.Linear(8024, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )

        self._feature_blocks = nn.ModuleList([
            conv1,
            res1_1,
            res2_1,
            res3_1,
            res1,
            pool1,
            conv2,
            pool2,
            conv3,
            res1_2,
            res2_2,
            res3_2,
            res2,
            conv4,
            conv5,
            pool5,
            fc_block,
        ])
        self.all_feat_names = [
            "conv1",
            "res1_1",
            "res2_1",
            "res3_1",
            "res1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "res1_2",
            "res2_2",
            "res3_2",
            "res2",
            "conv4",
            "conv5",
            "pool5",
            "fc_block",
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
        feat= x1 = self._feature_blocks[0](feat)  # cov1, 64x31x31
        key = self.all_feat_names[0]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[1](feat)  # res1_1, 64x31x31
        key = self.all_feat_names[1]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[2](feat)  # res1_2, 128x31x31
        key = self.all_feat_names[2]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[3](feat)  # res1_3, 64x31x31
        key = self.all_feat_names[3]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[4](feat + x1)  # res1, 64x31x31
        key = self.all_feat_names[4]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[5](feat)  # pool1, 64x15x15
        key = self.all_feat_names[5]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[6](feat)  # conv2, 192x15x15
        key = self.all_feat_names[6]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        pool2_ = feat = self._feature_blocks[7](feat)  # pool2, 192x7x7
        key = self.all_feat_names[7]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = x2 = self._feature_blocks[8](feat)  # conv3, 384x7x7
        key = self.all_feat_names[8]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[9](feat)  # res2_1, 256x7x7 without Relu but with batchnorm
        key = self.all_feat_names[9]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[10](feat)  # res2_2, 384x7x7 without Relu but with batchnorm
        key = self.all_feat_names[10]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[11](feat)  # res2_3, 384x7x7 without Relu but with batchnorm
        key = self.all_feat_names[11]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[12](feat + x2)  # res2, 384x7x7 without Relu but with batchnorm
        key = self.all_feat_names[12]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[13](feat)  # conv4, 256x7x7
        key = self.all_feat_names[13]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat_cat13 = torch.cat((feat, pool2_), 1)

        feat = self._feature_blocks[14](feat)  # conv5, 256x7x7
        key = self.all_feat_names[14]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[15](feat)  # pool5, 256x3x3
        key = self.all_feat_names[15]
        if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        feat = self._feature_blocks[16](feat)  # fcblock, --
        key = self.all_feat_names[16]
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

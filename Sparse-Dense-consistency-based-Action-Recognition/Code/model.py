# Model-file to be modified
import torch
import torchvision.models.video as video_models
from models.r2p1d import r2plus1d_18
from torch import nn
from torchsummary import summary


def build_r2plus1d_model(num_classes=38):
    model = r2plus1d_18(pretrained=False, progress=False)
    model.fc = nn.Linear(512, num_classes)
    # model = nn.Sequential(model, nn.Sigmoid())
    return model
    
def build_original_r2plus1d_model(num_classes=38, pretrained = False):
    model = video_models.r2plus1d_18(pretrained, progress=False)
    model.fc = nn.Linear(512, num_classes)
    # model = nn.Sequential(model, nn.Softmax())
    return model


if __name__ == '__main__':
    model = build_r2plus1d_model(num_classes = 102)
    # print(model)
    model.cuda()
    summary(model, (3, 16, 112, 112))
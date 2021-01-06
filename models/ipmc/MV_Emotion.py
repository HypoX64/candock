'''
Only works for project MV_Emotion
@Hypo
20201217
'''
import torch
from torch import nn
import torchvision

import sys
from ..net_2d import mobilenet,resnet,densenet

class BaseEncoder(nn.Module):
    def __init__(self, input_nc, encoder='resnet18'):
        super(BaseEncoder, self).__init__()

        if encoder == 'resnet18':
            self.net = resnet.resnet18(pretrained=True)
            self.net.conv1 = nn.Conv2d(input_nc, 64, 7, 2, 3, bias=False)
        elif encoder == 'mobilenet':
            self.net = mobilenet.mobilenet_v2(pretrained=True)
            self.net.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        elif encoder == 'densenet37':
            self.net = densenet.DenseNet(num_init_features=64, growth_rate=32, block_config=(3, 4, 6, 3))
            self.net.features.conv0 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif encoder == 'densenet121':
            self.net = densenet.DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16))
            self.net.features.conv0 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x

class SEweight(nn.Module):
    def __init__(self, input_nc, r=16):
        super(SEweight, self).__init__()

        self.weight=nn.Sequential(
            nn.Linear(input_nc, input_nc//r),
            nn.ReLU(inplace=True),
            nn.Linear(input_nc//r, input_nc),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.weight(x)
        x = x.view(x.size(0),-1)
        return x

class MV_Emotion(nn.Module):
    def __init__(self, num_classes, encoder = 'densenet121'):
        super(MV_Emotion, self).__init__()
        if encoder == 'resnet18':
            self.feature_num = 512
        elif encoder == 'mobilenet':
            self.feature_num = 1280
        elif encoder == 'densenet37':
            self.feature_num = 244
        elif encoder == 'densenet121':
            self.feature_num = 1024
        self.MicHeart  = BaseEncoder(1,encoder)
        self.MicBreath = BaseEncoder(2,encoder)
        self.PPGHeart  = BaseEncoder(2,encoder)
        self.SEweight1 = SEweight(self.feature_num)
        self.SEweight2 = SEweight(self.feature_num)
        self.SEweight3 = SEweight(self.feature_num)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_num*3, num_classes),
        )

    def forward(self, x):

        x_part1 = self.MicHeart((x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3))))
        x_part2 = self.MicBreath((x[:,1:3,:,:].view(x.size(0),2,x.size(2),x.size(3))))
        x_part3 = self.PPGHeart((x[:,3:5,:,:].view(x.size(0),2,x.size(2),x.size(3))))

        x_part1 = x_part1 * self.SEweight1(x_part1)
        x_part2 = x_part2 * self.SEweight2(x_part2)
        x_part3 = x_part3 * self.SEweight3(x_part3)

        x = torch.cat([x_part1,x_part2,x_part3],dim=1)
        x = self.classifier(x)

        return x


# class CNNweight(nn.Module):
#     def __init__(self, input_nc):
#         super(CNNweight, self).__init__()

#         self.weight = nn.Sequential(
#             nn.Conv2d(input_nc, 6, kernel_size=5, stride=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(6, 16, kernel_size=5, stride=3),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(16, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.weight(x)
#         x = x.view(x.size(0),-1)
#         return x

# class MV_Emotion(nn.Module):
#     def __init__(self, num_classes):
#         super(MV_Emotion, self).__init__()
#         self.MicHeart = BaseEncoder(1)
#         self.MicBreath = BaseEncoder(2)
#         self.PPGHeart = BaseEncoder(2)
#         self.CNNweight1 = CNNweight(1)
#         self.CNNweight2 = CNNweight(2)
#         self.CNNweight3 = CNNweight(2)

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(1280*3, num_classes),
#         )

#     def forward(self, x):
#         # x = self.MicHeart(x)
#         x_part1 = (x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3)))
#         x_part2 = (x[:,1:3,:,:].view(x.size(0),2,x.size(2),x.size(3)))
#         x_part3 = (x[:,3:5,:,:].view(x.size(0),2,x.size(2),x.size(3)))

#         weight1 = self.CNNweight1(x_part1)
#         weight2 = self.CNNweight2(x_part2)
#         weight3 = self.CNNweight3(x_part3)

#         x_part1 = self.MicHeart(x_part1)
#         x_part2 = self.MicBreath(x_part2)
#         x_part3 = self.PPGHeart(x_part3)


#         x_part1 = x_part1 + x_part1 * weight1.view(weight1.size(0),-1)
#         x_part2 = x_part2 + x_part2 * weight2.view(weight2.size(0),-1)
#         x_part3 = x_part3 + x_part3 * weight3.view(weight3.size(0),-1)


#         x = torch.cat([x_part1,x_part2,x_part3],dim=1)
#         x = self.classifier(x)

#         return x


# class MV_Emotion(nn.Module):
#     def __init__(self, num_classes):
#         super(MV_Emotion, self).__init__()
#         self.MicHeart = BaseEncoder(1)
#         self.MicBreath = BaseEncoder(2)
#         self.PPGHeart = BaseEncoder(2)
#         self.SEweight1 = SEweight(1280)
#         self.SEweight2 = SEweight(1280)
#         self.SEweight3 = SEweight(1280)

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(1280*3, num_classes),
#         )
#     def forward(self, x):
#         # x = self.MicHeart(x)
#         x_part1 = (x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3)))
#         x_part2 = (x[:,1:3,:,:].view(x.size(0),2,x.size(2),x.size(3)))
#         x_part3 = (x[:,3:5,:,:].view(x.size(0),2,x.size(2),x.size(3)))

#         x_part1 = self.MicHeart(x_part1)
#         x_part2 = self.MicBreath(x_part2)
#         x_part3 = self.PPGHeart(x_part3)

#         weight1 = self.SEweight1(x_part1)
#         weight2 = self.SEweight2(x_part2)
#         weight3 = self.SEweight3(x_part3)

#         x_part1 = x_part1 + x_part1 * weight1.view(weight1.size(0),-1)
#         x_part2 = x_part2 + x_part2 * weight2.view(weight2.size(0),-1)
#         x_part3 = x_part3 + x_part3 * weight3.view(weight3.size(0),-1)

#         # x_part1 = x_part1.mul(weight1.view(weight1.size(0),-1))
#         # x_part2 = x_part2.mul(weight2.view(weight2.size(0),-1))
#         # x_part3 = x_part3.mul(weight3.view(weight3.size(0),-1))

#         x = torch.cat([x_part1,x_part2,x_part3],dim=1)
#         x = self.classifier(x)

#         return x
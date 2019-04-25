import torch
from torch import nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel,kernel_size,stride=2):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.conv = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(outchannel)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.stride != 1:
            out += self.shortcut(x)
        else:
            out += x
        out = F.relu(out,inplace=True)
        return out

class Route(nn.Module):
    def __init__(self, kernel_size):
        super(Route, self).__init__()
        self.block1 = ResidualBlock(64, 64, kernel_size, stride=1)
        self.block2 = ResidualBlock(64, 128, kernel_size)
        self.block3 = ResidualBlock(128, 256, kernel_size)
        self.block4 = ResidualBlock(256, 512, kernel_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        return x

class Multi_Scale_ResNet(nn.Module):
    def __init__(self, inchannel, num_classes):
        super(Multi_Scale_ResNet, self).__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv1d(inchannel, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.Route1 = Route(3)
        self.Route2 = Route(5)
        self.Route3 = Route(7)
        self.fc = nn.Linear(512*3, num_classes)

    def forward(self, x):
        x = self.pre_conv(x)
        x1 = self.Route1(x)
        x2 = self.Route2(x)
        x3 = self.Route3(x)
        x = torch.cat((x1,x2,x3), 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
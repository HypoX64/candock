import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

def CreatNet(name):
    if name =='LSTM':
        return LSTM(100,27*5,5)
    elif name == 'CNN':
        return CNN()
    elif name == 'resnet18_1d':
        return resnet18_1d()
    elif name == 'dfcnn':
        return dfcnn()
    elif name in ['resnet101','resnet50','resnet18']:
        if name =='resnet101':
            net = torchvision.models.resnet101(pretrained=False)
            net.fc = nn.Linear(2048, 5)
        elif name =='resnet50':
            net = torchvision.models.resnet50(pretrained=False)
            net.fc = nn.Linear(2048, 5)
        elif name =='resnet18':
            net = torchvision.models.resnet18(pretrained=False)
            net.fc = nn.Linear(512, 5)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
             
        return net
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = torchvision.models.densenet121(pretrained=False)
        elif name == 'densenet201':
            net = torchvision.models.densenet201(pretrained=False)
        net.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        net.classifier = nn.Linear(4096, 5)
        return net


class LSTM(nn.Module):
    def __init__(self,INPUT_SIZE,TIME_STEP,CLASS_NUM,Hidden_size=256,Num_layers=2):
        super(LSTM, self).__init__()
        self.INPUT_SIZE=INPUT_SIZE
        self.TIME_STEP=TIME_STEP

        self.bn = nn.BatchNorm1d(INPUT_SIZE*TIME_STEP) 
        self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=Hidden_size,         # rnn hidden unit
            num_layers=Num_layers,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # self.dropout=nn.Dropout(0.5)

        self.out = nn.Linear(Hidden_size, CLASS_NUM)

    def forward(self, x):
        # x=self.bn(x)
        x=x.view(-1, self.TIME_STEP, self.INPUT_SIZE)
        r_out, (h_n, h_c) = self.lstm(x, None)   # None represents zero initial hidden state
        x=r_out[:, -1, :]
        # x=F.dropout(x,training=self.training)
        out = self.out(x)
        return out

class dfcnn(nn.Module):
    def __init__(self):
        super(dfcnn, self).__init__()
        self.layer1 = nn.Sequential(       
            nn.Conv2d(1, 32, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),   
        )
        self.layer2 = nn.Sequential(         
            nn.Conv2d(32, 64, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),             
            nn.MaxPool2d(2),                
        )
        self.layer3 = nn.Sequential(         
            nn.Conv2d(64, 128, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),             
            nn.MaxPool2d(2),                
        )
        self.layer4 = nn.Sequential(         
            nn.Conv2d(128, 256, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),                        
        )
        self.layer5 = nn.Sequential(         
            nn.Conv2d(256, 512, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, (3,3), 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),                        
        )

        self.out = nn.Linear(512*7*7, 5)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)
        # x = F.avg_pool1d(x, 175)
        x = x.view(x.size(0), -1)           
        output = self.out(x)
        return output

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(1, 64, 7, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace = True),                    
            nn.MaxPool1d(2),   
        )
        self.conv2 = nn.Sequential(         
            nn.Conv1d(64, 128, 7, 1, 0, bias=False),
            nn.BatchNorm1d(128),    
            nn.ReLU(inplace = True),                  
            nn.MaxPool1d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv1d(128, 256, 7, 1, 0, bias=False),
            nn.BatchNorm1d(256),    
            nn.ReLU(inplace = True),                     
            nn.MaxPool1d(2),               
        )
        self.conv4 = nn.Sequential(         
            nn.Conv1d(256, 512, 7, 1, 0, bias=False),
            nn.BatchNorm1d(512),    
            nn.ReLU(inplace = True),                     
            nn.MaxPool1d(2),               
        )
        self.conv5 = nn.Sequential(         
            nn.Conv1d(512, 1024, 7, 1, 0, bias=False),
            nn.BatchNorm1d(1024),    
            nn.ReLU(inplace = True),                                   
        )

        self.out = nn.Linear(1024, 5)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.avg_pool1d(x, 175)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outchannel)
            )

    def forward(self, x):
        identity = x
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=5):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=13, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool1d(out, 85)
 #       print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def resnet18_1d():
    return ResNet(ResidualBlock,5)
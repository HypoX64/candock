'''
@inproceedings{ganin2015unsupervised,
  title={Unsupervised domain adaptation by backpropagation},
  author={Ganin, Yaroslav and Lempitsky, Victor},
  booktitle={International conference on machine learning},
  pages={1180--1189},
  year={2015},
  organization={PMLR}
}
'''
import torch
from torch import nn
from .functions import ReverseLayerF
from ..net_2d import mobilenet,lightcnn,resnet,densenet
from ..ipmc import MV_Emotion


class Encoder(nn.Module):
    def __init__(self, input_nc, encoder='resnet18',avg_pool=True):
        super(Encoder, self).__init__()
        self.avg_pool = avg_pool

        if encoder == 'light':
            self.net = lightcnn.LightCNN(input_nc)
        elif encoder == 'resnet18':
            self.net = resnet.resnet18(pretrained=True)
            self.net.conv1 = nn.Conv2d(input_nc, 64, 7, 2, 3, bias=False)
            self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))
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
        # print(x.shape)
        if self.avg_pool:
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        else:
            x = x.reshape(x.shape[0], -1)
        return x

class ClassClassifier(nn.Module):
    def __init__(self,output_nc,feature_num):
        super(ClassClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(100, output_nc),
            nn.LogSoftmax(),
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class DomainClassifier(nn.Module):
    def __init__(self,feature_num,domain_num):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feature_num, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(100, domain_num),
            nn.LogSoftmax(dim=1),
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DANN(nn.Module):
    def __init__(self,input_nc,output_nc,domain_num,encoder='resnet18',avg_pool=False):
        super(DANN, self).__init__()
        if encoder == 'light':
            if avg_pool:
                self.feature_num = 128
            else:
                self.feature_num = 2048 # only for 28*28
        elif encoder == 'resnet18':
            if avg_pool:
                self.feature_num = 512
            else:
                self.feature_num = 512*9*8 # only for 257*251
        elif encoder == 'mobilenet':
            if avg_pool:
                self.feature_num = 1280
            else:
                self.feature_num = 1280*9*8 # only for 257*251
        elif encoder == 'densenet37':
            self.feature_num = 244
        elif encoder == 'densenet121':
            if avg_pool:
                self.feature_num = 1024
            else:
                self.feature_num = 1024*7*8 # only for 257*251
                #self.feature_num = 1024*4*8 # only for 257*126
        self.encoder = Encoder(input_nc,encoder,avg_pool)
        self.class_classifier = ClassClassifier(output_nc,self.feature_num)
        self.domain_classifier = DomainClassifier(self.feature_num,domain_num)

    def forward(self, x, alpha):
        
        feature = self.encoder(x)
        feature_reverse = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature_reverse)

        return class_output, domain_output
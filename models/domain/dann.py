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
from ..net_2d import mobilenet,lightcnn
from ..ipmc import MV_Emotion

class Encoder(nn.Module):
    def __init__(self, input_nc):
        super(Encoder, self).__init__()

        self.net = lightcnn.LightCNN(input_nc)

        # self.net = mobilenet.mobilenet_v2(pretrained=True)
        # self.net.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # self.net = densenet.densenet121(pretrained=False,num_classes = 1000)
        # self.net.features.conv0 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # print(x.size())
        return x

class ClassClassifier(nn.Module):
    def __init__(self,output_nc):
        super(ClassClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 100),
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
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DANN(nn.Module):
    def __init__(self,input_nc,output_nc):
        super(DANN, self).__init__()
        self.encoder = Encoder(input_nc)
        self.class_classifier = ClassClassifier(output_nc)
        self.domain_classifier = DomainClassifier()

    def forward(self, x, alpha):
        
        feature = self.encoder(x)
        feature_reverse = ReverseLayerF.apply(feature, alpha)

        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature_reverse)

        return class_output, domain_output
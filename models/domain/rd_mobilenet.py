'''
@inproceedings{hwang2020subject,
  title={Subject-Independent EEG-based Emotion Recognition using Adversarial Learning},
  author={Hwang, Sunhee and Ki, Minsong and Hong, Kibeom and Byun, Hyeran},
  booktitle={2020 8th International Winter Conference on Brain-Computer Interface (BCI)},
  pages={1--4},
  year={2020},
  organization={IEEE}
}
'''
import torch
from torch import nn
from . import mobilenet

class Encoder(nn.Module):
    def __init__(self,input_nc,feature_num):
        super(Encoder, self).__init__()
        self.trunk = mobilenet.mobilenet_v2(pretrained=True)
        self.trunk.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.trunk.classifier[1] = nn.Linear(in_features=1280, out_features=feature_num, bias=True)

    def forward(self, x):
        # print(x.size())
        x = self.trunk(x)
        return x

class ClassClassifier(nn.Module):
    def __init__(self,feature_num,output_nc):
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
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            )
        self.fc3 = nn.Sequential(
            nn.Linear(100, domain_num),
            nn.LogSoftmax(),
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self,input_nc,output_nc,feature_num,domain_num):
        super(Net, self).__init__()
        self.encoder = Encoder(input_nc, feature_num)
        self.class_classifier = ClassClassifier(feature_num, output_nc)
        self.domain_classifier = DomainClassifier(feature_num, domain_num)

    def forward(self, x, alpha):
        
        feature = self.encoder(x)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output
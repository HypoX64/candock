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
    def __init__(self,encoder,num_classes,domain_num,feature_num,avg_pool=False):
        super(DANN, self).__init__()
        self.avg_pool_flag = avg_pool
        self.encoder = encoder
        self.class_classifier = ClassClassifier(feature_num,num_classes)
        self.domain_classifier = DomainClassifier(feature_num,domain_num)

    def forward(self, x, alpha):
        
        feature = self.encoder(x)
        if self.avg_pool_flag:
            feature = nn.functional.adaptive_avg_pool2d(feature, 1).reshape(feature.shape[0], -1)
        else:
            feature = feature.reshape(feature.shape[0], -1)

        feature_reverse = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature_reverse)

        return class_output, domain_output
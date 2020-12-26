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
from ..net_2d import mobilenet,densenet

class BaseEncoder(nn.Module):
    def __init__(self, input_nc):
        super(BaseEncoder, self).__init__()

        self.net = mobilenet.mobilenet_v2(pretrained=True)
        self.net.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))
        # self.net = densenet.densenet121(pretrained=False,num_classes = 1000)
        # self.net.features.conv0 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        # self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        # print(x.size())
        return x

class RDNet(nn.Module):
    def __init__(self,input_nc,class_num,domain_num):
        super(RDNet, self).__init__()
        self.encoder = BaseEncoder(input_nc)
        # self.class_classifier = nn.Linear(1280, class_num)
        # self.domain_classifier = nn.Linear(1280, domain_num)
        self.class_classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024,class_num),
          )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024,domain_num),
          )

    def forward(self, x):
        
        feature = self.encoder(x)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature)

        return class_output, domain_output
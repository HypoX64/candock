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
from ..net_1d import lstm


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

class lstm_block(nn.Module):
    def __init__(self,input_size,time_step,Hidden_size=128,Num_layers=2):
        super(lstm_block, self).__init__()
        self.input_size=input_size
        self.time_step=time_step

        self.lstm = nn.LSTM(         
            input_size=input_size,
            hidden_size=Hidden_size,        
            num_layers=Num_layers,          
            batch_first=True, 
            # bidirectional = True      
        )

    def forward(self, x):
        # print(x.size())
        x=x.view(-1, self.time_step, self.input_size)
        r_out, (h_n, h_c) = self.lstm(x, None)  
        x=r_out[:, -1, :]
        return x

class lstm(nn.Module):
    def __init__(self,input_size,time_step,input_nc,Hidden_size=128,Num_layers=2):
        super(lstm, self).__init__()
        self.input_size=input_size
        self.time_step=time_step
        self.input_nc=input_nc
        self.point = input_size*time_step
       
        self.lstms = []
        for i in range(input_nc):
            exec('self.lstm'+str(i) + '=lstm_block(self.input_size,self.time_step,Hidden_size,Num_layers)')
            exec('self.lstms.append(self.lstm'+str(i)+')')
        # self.weight = SEweight(Hidden_size*input_nc)

    def forward(self, x):
        y = []
        for i in range(self.input_nc):
            y.append(self.lstms[i](x[:,i,:]))
        x = torch.cat(tuple(y), 1)
        # x = x*self.weight(x)
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
    def __init__(self,input_size, time_step, input_nc, output_nc, domain_num):
        super(DANN, self).__init__()
        self.feature_num = input_nc*128
        self.encoder = lstm(input_size, time_step, input_nc)
        self.class_classifier = ClassClassifier(self.feature_num,output_nc)
        self.domain_classifier = DomainClassifier(self.feature_num,domain_num)

    def forward(self, x, alpha):
        # print(x.size())
        feature = self.encoder(x)
        feature_reverse = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(feature_reverse)

        return class_output, domain_output
import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self,input_size,time_step,input_nc,num_classes,Hidden_size=128,Num_layers=2):
        super(lstm, self).__init__()
        self.input_size=input_size
        self.time_step=time_step
        self.input_nc=input_nc
        self.point = input_size*time_step
       
        self.lstms = []
        for i in range(input_nc):
            exec('self.lstm'+str(i) + '=lstm_block(self.input_size,self.time_step,Hidden_size,Num_layers)')
            exec('self.lstms.append(self.lstm'+str(i)+')')
        self.weight = SEweight(Hidden_size*input_nc)
        self.fc = self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(Hidden_size*input_nc, num_classes),
        )

    def forward(self, x):
        y = []
        for i in range(self.input_nc):
            y.append(self.lstms[i](x[:,i,:]))
        x = torch.cat(tuple(y), 1)
        x = x*self.weight(x)
        x = self.fc(x)
        return x
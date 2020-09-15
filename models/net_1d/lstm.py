import torch
from torch import nn
import torch.nn.functional as F

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
       
        for i in range(input_nc):
            exec('self.lstm'+str(i) + '=lstm_block(input_size, time_step, '+str(Hidden_size)+','+str(Num_layers)+')')
        self.fc = nn.Linear(Hidden_size*input_nc, num_classes)

    def forward(self, x):
        y = []
        x = x.view(-1,self.input_nc,self.time_step,self.input_size)
        for i in range(self.input_nc):
            y.append(eval('self.lstm'+str(i)+'(x[:,i,:])'))
        x = torch.cat(tuple(y), 1)
        # print(x.size())
        x = self.fc(x)
        return x
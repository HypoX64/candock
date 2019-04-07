import torch
from torch import nn
import torch.nn.functional as F

class lstm_block(nn.Module):
    def __init__(self,INPUT_SIZE,TIME_STEP,Hidden_size=128,Num_layers=2):
        super(lstm_block, self).__init__()
        self.INPUT_SIZE=INPUT_SIZE
        self.TIME_STEP=TIME_STEP

        self.lstm = nn.LSTM(         
            input_size=INPUT_SIZE,
            hidden_size=Hidden_size,        
            num_layers=Num_layers,          
            batch_first=True,       
        )

    def forward(self, x):
        x=x.view(-1, self.TIME_STEP, self.INPUT_SIZE)
        r_out, (h_n, h_c) = self.lstm(x, None)  
        x=r_out[:, -1, :]
        return x

class lstm(nn.Module):
    def __init__(self,INPUT_SIZE,TIME_STEP,num_classes,Hidden_size=128,Num_layers=2):
        super(lstm, self).__init__()
        self.INPUT_SIZE=INPUT_SIZE
        self.TIME_STEP=TIME_STEP
        self.point = INPUT_SIZE*TIME_STEP

        self.lstm1 = lstm_block(INPUT_SIZE, TIME_STEP)
        self.lstm2 = lstm_block(INPUT_SIZE, TIME_STEP)
        self.lstm3 = lstm_block(INPUT_SIZE, TIME_STEP)
        self.lstm4 = lstm_block(INPUT_SIZE, TIME_STEP)
        self.lstm5 = lstm_block(INPUT_SIZE, TIME_STEP)
        self.fc = nn.Linear(Hidden_size*5, num_classes)

    def forward(self, x):
        y = []
        for i in range(5):
            y.append(x[:,self.point*i:self.point*(i+1)].view(-1, self.TIME_STEP, self.INPUT_SIZE))
        y1 = self.lstm1(y[0])
        y2 = self.lstm2(y[1])
        y3 = self.lstm3(y[2])
        y4 = self.lstm4(y[3])
        y5 = self.lstm5(y[4])
        x = torch.cat((y1,y2,y3,y4,y5), 1) 
        x = self.fc(x)
        return x
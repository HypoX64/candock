import torch
from torch import nn
import torch.nn.functional as F
class cnn(nn.Module):
    def __init__(self, inchannel, num_classes):
        super(cnn, self).__init__()
        self.conv1 = nn.Sequential(       
            nn.Conv1d(inchannel, 64, 7, 1, 0, bias=False),
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
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(1024, num_classes)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)      
        x = self.out(x)
        return x
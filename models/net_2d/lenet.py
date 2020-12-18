import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module): 

    def __init__(self, input_nc=3, num_classes=10, flatten_num = 120 ):
        super(LeNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_nc, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(flatten_num, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        
        return x

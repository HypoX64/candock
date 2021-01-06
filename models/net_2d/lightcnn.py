import torch
from torch import nn
import torch.nn.functional as F

class LightCNN(nn.Module): 

    def __init__(self, input_nc=3, num_classes=10 ):
        super(LightCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

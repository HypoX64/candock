import torch
from torch import nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self, input_nc,num_classes,datasize):
        super(mlp, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(datasize*input_nc, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.net(x)
        return x
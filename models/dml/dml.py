import torch
from torch import nn

class DML(nn.Module):
    def __init__(self,encoder,n_feature,n_embedding):
        super(DML, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(n_feature, n_embedding)

    def forward(self, x):
        x = self.encoder(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
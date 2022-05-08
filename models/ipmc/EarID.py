'''
Only works for project EarID
@Hypo
20201217
'''
import torch
from torch import nn
import torchvision

from ..net_2d import resnet_cbam


class WearEncoder(nn.Module):
    def __init__(self, input_nc):
        super(WearEncoder, self).__init__()

        self.encoder = resnet_cbam.resnet18_cbam(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(input_nc, 64, 7, 2, 3, bias=False)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))

    def forward(self, x):
        x = self.encoder(x)
        return x

class HeartEncoder(nn.Module):
    def __init__(self, input_nc):
        super(HeartEncoder, self).__init__()

        self.encoder = resnet_cbam.resnet18_cbam(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(input_nc, 64, 7, 2, 3, bias=False)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-2]))

    def forward(self, x):
        x = self.encoder(x)
        return x

# test1
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EarID(nn.Module):
    def __init__(self, num_classes):
        super(EarID, self).__init__()
        self.WearEncoder   = WearEncoder(1)
        self.HeartEncoder  = HeartEncoder(1)
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):

        x_wear = self.WearEncoder((x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3))))
        x_heart = self.HeartEncoder((x[:,1,:,:].view(x.size(0),1,x.size(2),x.size(3))))

        x_wear = self.ca(x_heart) * x_wear
        x_wear = self.sa(x_heart) * x_wear
        x = torch.cat([x_wear,x_heart],dim=1)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# # test2
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return out

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x

# class WeightAttention(nn.Module):
#     def __init__(self, in_planes):
#         super(WeightAttention, self).__init__()
#         self.ca = ChannelAttention(in_planes)
#         self.sa = SpatialAttention()
#         self.sa_pool = nn.AdaptiveAvgPool2d(1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         w1 = self.ca(x)
#         w2 = self.sa(x)
#         w1 = torch.mean(w1,dim=1,keepdim=True)
#         w2 = self.sa_pool(w2)
#         w = w1+w2
#         w = self.sigmoid(w.reshape(w.shape[0],-1))
#         return w


# class EarID(nn.Module):
#     def __init__(self, num_classes):
#         super(EarID, self).__init__()
#         self.WearEncoder   = WearEncoder(1)
#         self.HeartEncoder  = HeartEncoder(1)
#         self.WearWeight = WeightAttention(512)
#         self.HeartWeight = WeightAttention(512)


#         self.fc = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.Linear(1024, num_classes),
#         )

#     def forward(self, x):
        
#         x_wear = self.WearEncoder((x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3))))
#         x_heart = self.HeartEncoder((x[:,1,:,:].view(x.size(0),1,x.size(2),x.size(3))))

#         w_wear = self.WearWeight(x_wear)
#         w_heart = self.HeartWeight(x_heart)

#         x_wear = nn.functional.adaptive_avg_pool2d(x_wear, 1).reshape(x_wear.shape[0], -1)
#         x_heart = nn.functional.adaptive_avg_pool2d(x_heart, 1).reshape(x_heart.shape[0], -1)

#         x_wear = x_wear*w_wear
#         x_heart = x_heart + x_heart*w_heart
#         x = torch.cat([x_wear,x_heart],dim=1)

#         x = self.fc(x)
#         return x
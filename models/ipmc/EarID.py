'''
Only works for project EarID
@Hypo
20201217
'''
import torch
from torch import nn
import torchvision

from . import mobilenet


class WearEncoder(nn.Module):
    def __init__(self, input_nc):
        super(BaseEncoder, self).__init__()

        self.lenet = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=5, stride=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.lenet(x)
        return x

class HeartEncoder(nn.Module):
    def __init__(self, input_nc):
        super(BaseEncoder, self).__init__()

        self.net = mobilenet.mobilenet_v2(pretrained=True)
        self.net.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))

    def forward(self, x):
        x = self.net(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x

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

class EarID(nn.Module):
    def __init__(self, num_classes):
        super(EarID, self).__init__()
        self.WearEncoder   = WearEncoder(1)
        self.HeartEncoder  = HeartEncoder(1)
        self.WearWeight    = SEweight(64)
        self.HeartWeight   = SEweight(1280)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280+64, num_classes),
        )
        
    def forward(self, x):

        x_wear = self.MicHeart((x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3))))
        x_heart = self.MicBreath((x[:,1,:,:].view(x.size(0),1,x.size(2),x.size(3))))

        x_wear = x_wear + x_wear * self.WearWeight(x_wear)
        x_heart = x_heart + x_heart * self.HeartWeight(x_heart)

        x = torch.cat([x_wear,x_heart],dim=1)
        x = self.classifier(x)

        return x

# class Leg1(nn.Module):
#     def __init__(self, input_nc, flatten_num, feature):
#         super(Leg1, self).__init__()

#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_nc, 6, kernel_size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(6, 16, kernel_size=5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
            
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten()
        
#         )
#         # self.fc = nn.Sequential(
#         #     nn.Linear(flatten_num, feature)
#         # )

#     def forward(self, x):
#         x = self.cnn(x)
#         # print(x.size())
#         # x = self.fc(x)
#         return x

# class Leg2(nn.Module):
#     def __init__(self, input_nc,feature):
#         super(Leg2, self).__init__()

#         self.net = mobilenet.mobilenet_v2(pretrained=True)
#         self.net.features[0][0] = nn.Conv2d(input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         self.net.classifier[1] = nn.Linear(in_features=1280, out_features=feature, bias=True)

#     def forward(self, x):
#         x = self.net(x)
#         return x


# class EarID(nn.Module):
#     def __init__(self, num_classes):
#         super(EarID, self).__init__()
#         self.leg1 = Leg1(1, 64 ,64)
        
#         self.get_weight=nn.Sequential(
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#             )

#         # self.leg1 = Leg1(1, 13456 ,120)
#         self.leg2 = Leg2(1,512)

#         self.fc = nn.Sequential(
#             nn.Linear(512+64, num_classes),
#             # nn.Linear(128, num_classes),
#         )


#     def forward(self, x):
#         x_ch1 = (x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3)))
#         x_ch2 = (x[:,1,:,:].view(x.size(0),1,x.size(2),x.size(3)))
#         x_ch1 = self.leg1(x_ch1)
#         ch1_weight = self.get_weight(x_ch1)
#         x_ch1 = x_ch1.mul(ch1_weight.view(x_ch1.size(0),1))
#         # print(x_ch1.size())
#         #print(ch1_weight.size())
#         x_ch2 = self.leg2(x_ch2)
#         x = torch.cat([x_ch1,x_ch2],dim=1)
#         x = self.fc(x)
#         return x
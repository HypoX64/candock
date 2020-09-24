import torch
from torch import nn
import torch.nn.functional as F


# class Autoencoder(nn.Module):
#     def __init__(self, input_nc,num_feature,num_classes,datasize):
#         super(Autoencoder, self).__init__()
#         self.datasize = datasize
#         self.finesize = (1+datasize//1024)*1024
#         self.prepad = nn.ReflectionPad1d((self.finesize-self.datasize)//2)

#         # encoder
#         self.encoder = Multi_Scale_ResNet(input_nc, num_feature)
#         self.class_fc = nn.Linear(num_feature, num_classes)

#         # decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(num_feature, 512),
#             nn.Linear(512, datasize)
#         )

#     def forward(self, x):
#         #print(x.size())
#         x = self.prepad(x)
#         #print(x.size())
#         feature = self.encoder(x) 

#         x = self.decoder(feature)
#         #x = x[:,:,(self.finesize-self.datasize)//2:(self.finesize-self.datasize)//2+self.datasize]
#         #print(x.size())
#         return x,feature



# class Autoencoder(nn.Module):
#     def __init__(self, input_nc,num_feature,num_classes,datasize):
#         super(Autoencoder, self).__init__()
#         self.datasize = datasize
#         self.finesize = (1+datasize//1024)*1024
#         self.prepad = nn.ReflectionPad1d((self.finesize-self.datasize)//2)

#         # encoder
#         encoder = [nn.ReflectionPad1d(3),
#                  nn.Conv1d(input_nc, 64, kernel_size=7, padding=0, bias=False),
#                  nn.BatchNorm1d(64),
#                  nn.ReLU(True)]
#         n_downsampling = 4
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             encoder += [nn.Conv1d(64 * mult, 64 * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
#                         nn.BatchNorm1d(64 * mult * 2),
#                         nn.ReLU(True),
#                         nn.MaxPool1d(2)]
#         encoder += [nn.AvgPool1d(8)]
#         self.encoder = nn.Sequential(*encoder)

#         self.fc1 = nn.Linear(self.finesize//2, num_feature)
#         self.class_fc = nn.Linear(num_feature, num_classes)
#         self.fc2 = nn.Linear(num_feature,self.finesize//2)

#         # decoder
#         decoder = [nn.Upsample(scale_factor = 8, mode='nearest')]
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             decoder += [
#                         nn.Upsample(scale_factor = 2, mode='nearest'),
#                         nn.ConvTranspose1d(64 * mult, int(64 * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=False),
#                       nn.BatchNorm1d(int(64 * mult / 2)),
#                       nn.ReLU(True)]
#         decoder += [nn.ReflectionPad1d(3)]
#         decoder += [nn.Conv1d(64, input_nc, kernel_size=7, padding=0)]

#         self.decoder = nn.Sequential(*decoder)

#     def forward(self, x):
#         #print(x.size())
#         x = self.prepad(x)
#         #print(x.size())
#         x = self.encoder(x) 
#         #print(x.size())  
#         x = x.view(x.size(0), -1)
#         #print(x.size())
#         feature = self.fc1(x)
#         out_class = self.class_fc(feature)
#         #print(feature.size())
#         x = self.fc2(feature)
#         x = x.view(x.size(0), -1, 1)
#         #print(x.size())
#         x = self.decoder(x)
#         x = x[:,:,(self.finesize-self.datasize)//2:(self.finesize-self.datasize)//2+self.datasize]
#         #print(x.size())
#         return x,feature,out_class

class Autoencoder(nn.Module):
    def __init__(self, input_nc,num_feature,num_classes,datasize):
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(datasize, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, num_feature),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(num_feature, 12),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, datasize),
        )


    def forward(self, x):
        feature = self.encoder(x)
        x = self.decoder(feature)
        return x,feature
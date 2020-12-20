import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self,signal_size,output_nc,latent_dim):
        super(Generator, self).__init__()

        self.init_size = signal_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 31, stride=1, padding=15),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 31, stride=1, padding=15),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, output_nc, 31, stride=1, padding=15),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size)
        signal = self.conv_blocks(out)
        return signal


class Discriminator(nn.Module):
    def __init__(self,signal_size,input_nc):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv1d(in_filters, out_filters, 31, 2, 15), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = signal_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size, 1), nn.Sigmoid())

    def forward(self, signal):
        out = self.model(signal)
        # print(out.size())
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class GANloss(nn.Module):
    def __init__(self,gpu_id,batchsize):
        super(GANloss,self).__init__()
        self.Tensor = torch.cuda.FloatTensor if gpu_id != '-1' else torch.FloatTensor
        self.valid = Variable(self.Tensor(batchsize, 1).fill_(1.0), requires_grad=False)
        self.fake = Variable(self.Tensor(batchsize, 1).fill_(0.0), requires_grad=False)
        self.loss_function = torch.nn.BCELoss()
        
    def forward(self,tensor,target_is_real):
        if target_is_real:
            loss = self.loss_function(tensor,self.valid)
        else:
            loss = self.loss_function(tensor,self.fake)
        return loss
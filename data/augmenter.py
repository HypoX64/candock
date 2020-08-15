import os
import time

import numpy as np
import torch
from torch import nn, optim
from multiprocessing import Process, Queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
from util import util,transformer,dataloader,statistics,plot,options
from models.net_1d.gan import Generator,Discriminator,GANloss,weights_init_normal
from models.core import show_paramsnumber

def gan(opt,signals,labels):
    print('Augment dataset using gan...')
    if opt.gpu_id != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    if not opt.no_cudnn:
        torch.backends.cudnn.benchmark = True

    signals_train = signals[:opt.fold_index[0]]
    labels_train  = labels[:opt.fold_index[0]]
    signals_eval = signals[opt.fold_index[0]:]
    labels_eval  = labels[opt.fold_index[0]:]


    signals_train = signals_train[labels_train.argsort()]
    labels_train = labels_train[labels_train.argsort()]
    out_signals = signals_train.copy()
    out_labels = labels_train.copy()
    label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels_train)
    opt = options.get_auto_options(opt, signals_train, labels_train)


    generator = Generator(opt.loadsize,opt.input_nc,opt.gan_latent_dim)
    discriminator = Discriminator(opt.loadsize,opt.input_nc)
    show_paramsnumber(generator, opt)
    show_paramsnumber(discriminator, opt)

    ganloss = GANloss(opt.gpu_id,opt.batchsize)

    if opt.gpu_id != -1:
        generator.cuda()
        discriminator.cuda()
        ganloss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.gan_lr, betas=(0.5, 0.999))

    index_cnt = 0
    for which_label in range(len(label_cnt)):

        if which_label in opt.gan_labels:
            sub_signals = signals_train[index_cnt:index_cnt+label_cnt[which_label]]
            sub_labels = labels_train[index_cnt:index_cnt+label_cnt[which_label]]

            generator.apply(weights_init_normal)
            discriminator.apply(weights_init_normal)
            generator.train()
            discriminator.train()

            for epoch in range(opt.gan_epochs):
                epoch_g_loss = 0
                epoch_d_loss = 0
                iter_pre_epoch = len(sub_labels)//opt.batchsize
                transformer.shuffledata(sub_signals, sub_labels)
                t1 = time.time()
                for i in range(iter_pre_epoch):
                    real_signal = sub_signals[i*opt.batchsize:(i+1)*opt.batchsize].reshape(opt.batchsize,opt.input_nc,opt.loadsize)
                    real_signal = transformer.ToTensor(real_signal,gpu_id=opt.gpu_id)

                    #  Train Generator
                    optimizer_G.zero_grad()
                    z = transformer.ToTensor(np.random.normal(0, 1, (opt.batchsize, opt.gan_latent_dim)),gpu_id = opt.gpu_id)
                    gen_signal = generator(z)
                    g_loss = ganloss(discriminator(gen_signal),True)
                    epoch_g_loss += g_loss.item()
                    g_loss.backward()
                    optimizer_G.step()

                    #  Train Discriminator
                    optimizer_D.zero_grad()
                    d_real = ganloss(discriminator(real_signal), True)
                    d_fake = ganloss(discriminator(gen_signal.detach()), False)
                    d_loss = (d_real + d_fake) / 2
                    epoch_d_loss += d_loss.item()
                    d_loss.backward()
                    optimizer_D.step()
                t2 = time.time()
                print(
                    "[Label %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [time: %.2f]"
                    % (sub_labels[0], epoch+1, opt.gan_epochs, epoch_g_loss/iter_pre_epoch, epoch_d_loss/iter_pre_epoch, t2-t1)
                )

            plot.draw_gan_result(real_signal.data.cpu().numpy(), gen_signal.data.cpu().numpy(),opt)

            generator.eval()
            for i in range(int(len(sub_labels)*(opt.gan_augment_times-1))//opt.batchsize):
                z = transformer.ToTensor(np.random.normal(0, 1, (opt.batchsize, opt.gan_latent_dim)),gpu_id = opt.gpu_id)
                gen_signal = generator(z)
                out_signals = np.concatenate((out_signals, gen_signal.data.cpu().numpy()))
                #print(np.ones((opt.batchsize),dtype=np.int64)*which_label)
                out_labels = np.concatenate((out_labels,np.ones((opt.batchsize),dtype=np.int64)*which_label))

        index_cnt += label_cnt[which_label]
    opt.fold_index = [len(out_labels)]
    out_signals = np.concatenate((out_signals, signals_eval))
    out_labels = np.concatenate((out_labels, labels_eval))
    # return signals,labels
    return out_signals,out_labels


def base(opt,signals,labels):
    pass

def augment(opt,signals,labels):
    pass

if __name__ == '__main__':

    opt = options.Options().getparse()
    signals,labels = dataloader.loaddataset(opt)
    out_signals,out_labels = gan(opt,signals,labels,2)
    print(out_signals.shape,out_labels.shape)
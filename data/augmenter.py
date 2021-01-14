import os
import time

import numpy as np
import scipy.signal
import scipy.fftpack as fftpack
import pywt

import torch
from torch import nn, optim
from multiprocessing import Process, Queue
import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
from util import util,plot,options,dsp
from util import array_operation as arr
from . import transforms,dataloader,statistics,surrogates,noise

from models.net_1d.gan import Generator,Discriminator,GANloss,weights_init_normal
from core import show_paramsnumber

def dcgan(opt,signals,labels):
    print('Augment dataset using gan...')
    if opt.gpu_id != '-1':
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

    if opt.gpu_id != '-1':
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

def base1d(opt,data,test_flag):
    """
    data : batchsize,ch,length
    """
    batchsize,ch,length = data.shape
    random_list = np.random.rand(15)
    threshold = 1/(len(opt.augment)+1)
    noise_lambda = opt.augment_noise_lambda
    if test_flag:
        move = int((length-opt.finesize)*0.5)
        result = data[:,:,move:move+opt.finesize]
    else:
        result = np.zeros((batchsize,ch,opt.finesize))
        
        for i in range(batchsize):
            for j in range(ch):
                signal = data[i][j]
                _length = length
                # Time Domain
                if 'scale' in opt.augment and random_list[0]>threshold:
                    beta = np.clip(np.random.normal(1, 0.1),0.8,1.2)
                    signal = arr.interp(signal, int(_length*beta))
                    _length = signal.shape[0]


                if 'warp' in opt.augment and random_list[1]>threshold:
                    pos = np.sort(np.random.randint(0, _length, 2))
                    if pos[1]-pos[0]>10:
                        beta = np.clip(np.random.normal(1, 0.1),0.8,1.2)
                        signal = np.concatenate((signal[:pos[0]], arr.interp(signal[pos[0]:pos[1]], int((pos[1]-pos[0])*beta)) , signal[pos[1]:]))
                        _length = signal.shape[0]

                # Noise            
                if 'spike' in opt.augment and random_list[2]>threshold:
                    std = np.std(signal)
                    spike_indexs = np.random.randint(0, _length, int(_length*np.clip(np.random.uniform(0,0.05),0,1)))
                    for index in spike_indexs:
                        signal[index] = signal[index] + std*np.random.randn()*noise_lambda
                
                if 'step' in opt.augment and random_list[3]>threshold:
                    std = np.std(signal)
                    step_indexs = np.random.randint(0, _length, int(_length*np.clip(np.random.uniform(0,0.01),0,1)))
                    for index in step_indexs:
                        signal[index:] = signal[index:] + std*np.random.randn()*noise_lambda
                
                if 'slope' in opt.augment and random_list[4]>threshold: 
                    slope = np.linspace(-1, 1, _length)*np.random.randn()
                    signal = signal+slope*noise_lambda

                if 'white' in opt.augment and random_list[5]>threshold:
                    signal = signal+noise.noise(_length,'white')*(np.std(signal)*np.random.randn()*noise_lambda)

                if 'pink' in opt.augment and random_list[6]>threshold:
                    signal = signal+noise.noise(_length,'pink')*(np.std(signal)*np.random.randn()*noise_lambda)

                if 'blue' in opt.augment and random_list[7]>threshold:
                    signal = signal+noise.noise(_length,'blue')*(np.std(signal)*np.random.randn()*noise_lambda)

                if 'brown' in opt.augment and random_list[8]>threshold:
                    signal = signal+noise.noise(_length,'brown')*(np.std(signal)*np.random.randn()*noise_lambda)

                if 'violet' in opt.augment and random_list[9]>threshold:
                    signal = signal+noise.noise(_length,'violet')*(np.std(signal)*np.random.randn()*noise_lambda)

                # Frequency Domain
                if 'app' in opt.augment and random_list[10]>threshold:
                    # amplitude and phase perturbations
                    signal = surrogates.app(signal)

                if 'aaft' in opt.augment and random_list[11]>threshold:  
                    # Amplitude Adjusted Fourier Transform
                    signal = surrogates.aaft(signal)

                if 'iaaft' in opt.augment and random_list[12]>threshold:
                    # Iterative Amplitude Adjusted Fourier Transform
                    signal = surrogates.iaaft(signal,10)[0]

                # crop and filp
                if 'filp' in opt.augment and random_list[13]>threshold:
                    signal = signal[::-1]

                if _length >= opt.finesize:
                    move = int((_length-opt.finesize)*np.random.random())
                    signal = signal[move:move+opt.finesize]
                else:
                    signal = arr.pad(signal, opt.finesize-_length, mod = 'repeat')

                result[i,j] = signal
    return result

def base2d(img,finesize = (224,244),test_flag = True):
    h,w = img.shape[:2]
    if test_flag:
        h_move = int((h-finesize[0])*0.5)
        w_move = int((w-finesize[1])*0.5)
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
    else:
        #random crop
        h_move = int((h-finesize[0])*random.random())
        w_move = int((w-finesize[1])*random.random())
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
        #random flip
        if random.random()<0.5:
            result = result[:,::-1]
        #random amp
        result = result*random.uniform(0.9,1.1)+random.uniform(-0.05,0.05)
    return result

def augment(opt,signals,labels):
    pass

if __name__ == '__main__':

    opt = options.Options().getparse()
    signals,labels = dataloader.loaddataset(opt)
    out_signals,out_labels = gan(opt,signals,labels,2)
    print(out_signals.shape,out_labels.shape)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import util
import transformer
import dataloader
from options import Options
from creatnet import CreatNet

'''
--------------------------------preload data--------------------------------
@hypox64
2020/04/03
'''
opt = Options().getparse()
net = CreatNet(opt)

#load data
signals = np.load('./datasets/simple_test/signals.npy')
labels = np.load('./datasets/simple_test/labels.npy')

#load prtrained_model
net.load_state_dict(torch.load('./checkpoints/pretrained/micro_multi_scale_resnet_1d_50class.pth'))
net.eval()
if not opt.no_cuda:
    net.cuda()

for signal,true_label in zip(signals, labels):
    signal = signal.reshape(1,1,-1) #batchsize,ch,length
    true_label = true_label.reshape(1) #batchsize
    signal,true_label = transformer.ToTensor(signal,true_label,no_cuda =opt.no_cuda)
    out = net(signal)
    pred_label = torch.max(out, 1)[1]
    pred_label=pred_label.data.cpu().numpy()
    true_label=true_label.data.cpu().numpy()
    print(("true:{0:d} predict:{1:d}").format(true_label[0],pred_label[0]))

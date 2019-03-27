import argparse
import os
import numpy as np
import torch
#filedir = '/media/hypo/Hypo/physionet_org_train'
# filedir ='E:\physionet_org_train'

#'/media/hypo/Hypo/physionet_org_train'
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--no_cuda', action='store_true', help='if true, do not use gpu')
        self.parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
        self.parser.add_argument('--batchsize', type=int, default=16,help='batchsize')
        self.parser.add_argument('--dataset_dir', type=str, default='./sleep-edfx/sleep-cassette',
                                help='your dataset path')
        self.parser.add_argument('--dataset_name', type=str, default='sleep-edf',help='Choose dataset')
        self.parser.add_argument('--signal_name', type=str, default='EEG Fpz-Cz',help='Choose the EEG channel C4-M1|EEG Fpz-Cz')
        self.parser.add_argument('--sample_num', type=int, default=20,help='the amount you want to load')
        self.parser.add_argument('--model_name', type=str, default='resnet18',help='Choose model')
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--weight_mod', type=str, default='avg_best',help='Choose weight mode: avg_best|normal')

        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        if self.opt.dataset_name == 'sleep-edf':
            self.opt.sample_num = 8


        if self.opt.weight_mod == 'normal':
            weight = np.array([1,1,1,1,1])
        elif self.opt.weight_mod == 'avg_best':
            if self.opt.dataset_name == 'CinC_Challenge_2018':
                weight = np.log(1/np.array([0.15,0.3,0.08,0.13,0.18]))
            elif self.opt.dataset_name == 'sleep-edfx':
                weight = np.log(1/np.array([0.08,0.30,0.05,0.15,0.35]))
            elif self.opt.dataset_name == 'sleep-edf':
                weight = np.log(1/np.array([0.08,0.23,0.02,0.10,0.53]))
      
        self.opt.weight = weight


        return self.opt
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
        self.parser.add_argument('--dataset_dir', type=str, default='./sleep-edfx/sleep-telemetry',
                                help='your dataset path')
        self.parser.add_argument('--dataset_name', type=str, default='sleep-edfx',help='Choose dataset')
        self.parser.add_argument('--signal_name', type=str, default='C4-M1',help='Choose the EEG channel')
        self.parser.add_argument('--signal_num', type=int, default=44,help='the amount you want to load')
        self.parser.add_argument('--model_name', type=str, default='LSTM',help='Choose model')
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--weight_mod', type=str, default='normal',help='Choose weight mod')

        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.dataset_name == 'CinC_Challenge_2018':
            if self.opt.weight_mod == 'avg_best':
                weight = np.log(np.array([1/0.15,1/0.3,1/0.08,1/0.13,1/0.18]))
            elif self.opt.weight_mod == 'normal':
                weight = np.array([1,1,1,1,1])

        elif self.opt.dataset_name == 'sleep-edfx':
            if self.opt.weight_mod == 'avg_best':
                weight = np.log(1/np.array([0.08,0.30,0.05,0.15,0.35]))
            elif self.opt.weight_mod == 'normal':
                weight = np.array([1,1,1,1,1])        
        self.opt.weight = torch.from_numpy(weight).float()


        return self.opt
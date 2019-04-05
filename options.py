import argparse
import os
import numpy as np
import torch
#python3 train.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name CinC_Challenge_2018 --signal_name C4-M1 --sample_num 200 --model_name resnet18 --batchsize 32 --epochs 10 --fold_num 5 --pretrained
#python3 train_new.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name CinC_Challenge_2018 --signal_name C4-M1 --sample_num 10 --model_name LSTM --batchsize 32 --network_save_freq 100 --epochs 10
#python3 train.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name CinC_Challenge_2018 --signal_name C4-M1 --sample_num 10 --model_name resnet18 --batchsize 32
#filedir = '/media/hypo/Hypo/physionet_org_train'
# filedir ='E:\physionet_org_train'
#python3 train.py --dataset_name sleep-edf --model_name resnet50 --batchsize 4 --epochs 50 --pretrained
#'/media/hypo/Hypo/physionet_org_train'
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--no_cuda', action='store_true', help='if input, do not use gpu')
        self.parser.add_argument('--no_cudnn', action='store_true', help='if input, do not use cudnn')
        self.parser.add_argument('--pretrained', action='store_true', help='if input, use pretrained models')
        self.parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
        self.parser.add_argument('--fold_num', type=int, default=5,help='k-fold')
        self.parser.add_argument('--batchsize', type=int, default=16,help='batchsize')
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/sleep-edfx/',
                                help='your dataset path')
        self.parser.add_argument('--dataset_name', type=str, default='sleep-edf',help='Choose dataset sleep-edf|sleep-edf|CinC_Challenge_2018|')
        self.parser.add_argument('--select_sleep_time', action='store_true', help='if input, for sleep-cassette only use sleep time to train')
        self.parser.add_argument('--signal_name', type=str, default='EEG Fpz-Cz',help='Choose the EEG channel C4-M1|EEG Fpz-Cz')
        self.parser.add_argument('--sample_num', type=int, default=20,help='the amount you want to load')
        self.parser.add_argument('--model_name', type=str, default='resnet18',help='Choose model')
        self.parser.add_argument('--epochs', type=int, default=50,help='end epoch')
        self.parser.add_argument('--weight_mod', type=str, default='avg_best',help='Choose weight mode: avg_best|normal')
        self.parser.add_argument('--network_save_freq', type=int, default=5,help='the freq to save network')



        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.dataset_name == 'sleep-edf':
            self.opt.sample_num = 8


        # if self.opt.weight_mod == 'normal':
        #     weight = np.array([1,1,1,1,1])
        # elif self.opt.weight_mod == 'avg_best':
        #     if self.opt.dataset_name == 'CinC_Challenge_2018':
        #         weight = np.log(1/np.array([0.15,0.3,0.08,0.13,0.18]))
        #     elif self.opt.dataset_name == 'sleep-edfx':
        #         weight = np.log(1/np.array([0.04,0.20,0.04,0.08,0.63]))
        #     elif self.opt.dataset_name == 'sleep-edf':
        #         weight = np.log(1/np.array([0.08,0.23,0.01,0.10,0.53]))
        #         if self.opt.select_sleep_time:
        #             weight = np.log(1/np.array([0.16,0.44,0.05,0.19,0.53]))
      
        # self.opt.weight = weight


        return self.opt
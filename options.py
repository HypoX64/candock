import argparse
import os
import numpy as np
import torch

#python3 train.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name cc2018 --signal_name 'C4-M1' --sample_num 10 --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 20 --lr 0.0005 --BID 5_95_th --select_sleep_time --cross_validation subject
# python3 train.py --dataset_dir './datasets/sleep-edfx/' --dataset_name sleep-edfx --signal_name 'EEG Fpz-Cz' --sample_num 10 --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 20 --lr 0.0005 --BID 5_95_th --select_sleep_time --cross_validation subject

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--no_cuda', action='store_true', help='if input, do not use gpu')
        self.parser.add_argument('--no_cudnn', action='store_true', help='if input, do not use cudnn')
        self.parser.add_argument('--pretrained', action='store_true', help='if input, use pretrained models')
        self.parser.add_argument('--lr', type=float, default=0.0005,help='learning rate')
        self.parser.add_argument('--cross_validation', type=str, default='subject',help='k-fold | subject')
        self.parser.add_argument('--BID', type=str, default='5_95_th',help='Balance individualized differences  5_95_th | median |None')
        self.parser.add_argument('--fold_num', type=int, default=5,help='k-fold')
        self.parser.add_argument('--batchsize', type=int, default=64,help='batchsize')
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/sleep-edfx/',
                                help='your dataset path')
        self.parser.add_argument('--dataset_name', type=str, default='sleep-edfx',help='Choose dataset sleep-edfx | sleep-edf | cc2018')
        self.parser.add_argument('--select_sleep_time', action='store_true', help='if input, for sleep-cassette only use sleep time to train')
        self.parser.add_argument('--signal_name', type=str, default='EEG Fpz-Cz',help='Choose the EEG channel C4-M1 | EEG Fpz-Cz |...')
        self.parser.add_argument('--sample_num', type=int, default=10,help='the amount you want to load')
        self.parser.add_argument('--model_name', type=str, default='lstm',help='Choose model  lstm | multi_scale_resnet_1d | resnet18 |...')
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--weight_mod', type=str, default='avg_best',help='Choose weight mode: avg_best|normal')
        self.parser.add_argument('--network_save_freq', type=int, default=5,help='the freq to save network')

        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.dataset_name == 'sleep-edf':
            self.opt.sample_num = 8
        if self.opt.no_cuda:
            self.opt.no_cudnn = True
        if self.opt.fold_num == 0:
            self.opt.fold_num = 1
        if self.opt.cross_validation == 'subject':
            self.opt.fold_num = 1

        return self.opt
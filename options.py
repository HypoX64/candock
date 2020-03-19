import argparse
import os
import numpy as np
import torch

# python3 train.py --dataset_dir '/media/hypo/Hypo/physionet_org_train' --dataset_name cc2018 --signal_name 'C4-M1' --sample_num 20 --model_name lstm --batchsize 64 --epochs 20 --lr 0.0005 --no_cudnn
# python3 train.py --dataset_dir './datasets/sleep-edfx/' --dataset_name sleep-edfx --signal_name 'EEG Fpz-Cz' --sample_num 50  --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 25 --lr 0.0005 --BID 5_95_th --select_sleep_time --no_cudnn --select_sleep_time

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        #base
        self.parser.add_argument('--no_cuda', action='store_true', help='if input, do not use gpu')
        self.parser.add_argument('--no_cudnn', action='store_true', help='if input, do not use cudnn')
        self.parser.add_argument('--label', type=int, default=5,help='number of labels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input channels')
        self.parser.add_argument('--label_name', type=str, default='auto',help='name of labels,example:"a,b,c,d,e,f"')
        self.parser.add_argument('--model_name', type=str, default='lstm',help='Choose model  lstm | multi_scale_resnet_1d | resnet18 | micro_multi_scale_resnet_1d...')
        self.parser.add_argument('--pretrained', action='store_true', help='if input, use pretrained models')
        self.parser.add_argument('--continue_train', action='store_true', help='if input, continue train')
        self.parser.add_argument('--lr', type=float, default=0.001,help='learning rate') 
        self.parser.add_argument('--batchsize', type=int, default=64,help='batchsize')
        self.parser.add_argument('--weight_mod', type=str, default='normal',help='Choose weight mode: auto | normal')
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--network_save_freq', type=int, default=5,help='the freq to save network')
        self.parser.add_argument('--k_fold', type=int, default=0,help='fold_num of k-fold.if 0 or 1,no k-fold')
        
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/sleep-edfx/',
                                help='your dataset path')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints/',help='save checkpoints')
        self.parser.add_argument('--dataset_name', type=str, default='preload',
            help='Choose dataset preload | sleep-edfx | cc2018  ,preload:your data->shape:(num,ch,length), sleep-edfx&cc2018:sleep stage')
        self.parser.add_argument('--separated', action='store_true', help='for preload data, if input, load separated train and test datasets')
        self.parser.add_argument('--no_shuffle', action='store_true', help='do not shuffle data when load')


        #for EEG datasets  
        self.parser.add_argument('--BID', type=str, default='5_95_th',help='Balance individualized differences  5_95_th | median |None')
        self.parser.add_argument('--select_sleep_time', action='store_true', help='if input, for sleep-cassette only use sleep time to train')
        self.parser.add_argument('--signal_name', type=str, default='EEG Fpz-Cz',help='Choose the EEG channel C4-M1 | EEG Fpz-Cz |...')
        self.parser.add_argument('--sample_num', type=int, default=20,help='the amount you want to load')
        
        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.dataset_name == 'sleep-edf':
            self.opt.sample_num = 8
        if self.opt.no_cuda:
            self.opt.no_cudnn = True

        if self.opt.k_fold == 0 :
            self.opt.k_fold = 1

        if self.opt.label_name == 'auto':
            if self.opt.dataset_name in ['sleep-edf','sleep-edfx','cc2018']:
                self.opt.label_name = ["N3", "N2", "N1", "REM","W"]
            else:
                names = []
                for i in range(self.opt.label):
                    names.append(str(i))
                self.opt.label_name = names
        else:
            names = self.opt.label_name
            names = names.replace(" ", "")
            names = names.split(",")
            self.opt.label_name = names

        return self.opt
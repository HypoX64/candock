import os
import time
import random

import scipy.io as sio
import numpy as np

import sys
sys.path.append("..")
from . import transforms,statistics
from util import dsp
from util import array_operation as arr


def del_labels(signals,labels,dels):
    del_index = []
    for i in range(len(labels)):
        if labels[i] in dels:
            del_index.append(i)
    del_index = np.array(del_index)
    signals = np.delete(signals,del_index, axis = 0)
    labels = np.delete(labels,del_index,axis = 0)
    return signals,labels


def segment_traineval_dataset(signals,labels,a=0.8,random=True):
    length = len(labels)
    if random:
        transforms.shuffledata(signals, labels)
        signals_train = signals[:int(a*length)]
        labels_train = labels[:int(a*length)]
        signals_eval = signals[int(a*length):]
        labels_eval = labels[int(a*length):]
    else:
        label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
        #signals_train=[];labels_train=[];signals_eval=[];labels_eval=[]
        # cnt_ori = 0
        # signals_tmp=np.zeros_like(signals)
        # labels_tmp=np.zeros_like(labels)
        cnt = 0
        for i in range(label_num):
            if i ==0:
                signals_train = signals[cnt:cnt+int(label_cnt[i]*0.8)]
                labels_train =  labels[cnt:cnt+int(label_cnt[i]*0.8)]
                signals_eval =  signals[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]
                labels_eval =   labels[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]
            else:
                signals_train = np.concatenate((signals_train, signals[cnt:cnt+int(label_cnt[i]*0.8)]))
                labels_train = np.concatenate((labels_train, labels[cnt:cnt+int(label_cnt[i]*0.8)]))

                signals_eval = np.concatenate((signals_eval, signals[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]))
                labels_eval = np.concatenate((labels_eval, labels[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]))
            cnt += label_cnt[i]
    return signals_train,labels_train,signals_eval,labels_eval

def balance_label(signals,labels):

    label_sta,_,label_num = statistics.label_statistics(labels)
    ori_length = len(labels)
    max_label_length = max(label_sta)
    signals = signals[labels.argsort()]
    labels = labels[labels.argsort()]

    if signals.ndim == 2:
        new_signals = np.zeros((max_label_length*label_num,signals.shape[1]), dtype=signals.dtype)
    elif signals.ndim == 3:
        new_signals = np.zeros((max_label_length*label_num,signals.shape[1],signals.shape[2]), dtype=signals.dtype)
    new_labels = np.zeros((max_label_length*label_num), dtype=labels.dtype)
    new_signals[:ori_length] = signals
    new_labels[:ori_length] = labels
    del(signals)
    del(labels)

    cnt = ori_length
    for label in range(len(label_sta)):
        if label_sta[label] < max_label_length:
            if label == 0:
                start = 0
            else:
                start = np.sum(label_sta[:label])
            end = np.sum(label_sta[:label+1])-1

            for i in range(max_label_length-label_sta[label]):
                new_signals[cnt] = new_signals[random.randint(start,end)]
                new_labels[cnt] = label
                cnt +=1
    return new_signals,new_labels

#load all data in datasets
def loaddataset(opt): 
    print('Loading dataset...')

    signals = np.load(os.path.join(opt.dataset_dir,'signals.npy'))
    labels = np.load(os.path.join(opt.dataset_dir,'labels.npy'))
    num,ch,size = signals.shape

    # normliaze
    if opt.normliaze != 'None':
        for i in range(num):
            for j in range(ch):
                signals[i][j] = arr.normliaze(signals[i][j], mode = opt.normliaze, truncated=5)
    # filter
    if opt.filter != 'None':
        for i in range(num):
            for j in range(ch): 
                if opt.filter == 'fft':
                    signals[i][j] = dsp.fft_filter(signals[i][j], opt.filter_fs, opt.filter_fc,type = opt.filter_mod) 
                elif opt.filter == 'iir':         
                    signals[i][j] = dsp.bpf(signals[i][j], opt.filter_fs, opt.filter_fc[0], opt.filter_fc[1], numtaps=3, mode='iir')
                elif opt.filter == 'fir':
                    signals[i][j] = dsp.bpf(signals[i][j], opt.filter_fs, opt.filter_fc[0], opt.filter_fc[1], numtaps=101, mode='fir')
    
    # wave filter
    if opt.wave != 'None':
        for i in range(num):
            for j in range(ch):
                signals[i][j] = dsp.wave_filter(signals[i][j],opt.wave,opt.wave_level,opt.wave_usedcoeffs)
    
    # use fft to improve frequency domain information
    if opt.augment_fft:
        new_signals = np.zeros((num,ch*2,size), dtype=np.float32)
        new_signals[:,:ch,:] = signals
        for i in range(num):
            for j in range(ch):
                new_signals[i,ch+j,:] = dsp.fft(signals[i,j,:],half=False)
        signals = new_signals

    if opt.fold_index == 'auto':
        transforms.shuffledata(signals,labels)

    return signals.astype(np.float32),labels.astype(np.int64)
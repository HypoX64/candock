import os
import time
import random

import scipy.io as sio
import numpy as np

from . import dsp,transformer,statistics
from . import array_operation as arr


def del_labels(signals,labels,dels):
    del_index = []
    for i in range(len(labels)):
        if labels[i] in dels:
            del_index.append(i)
    del_index = np.array(del_index)
    signals = np.delete(signals,del_index, axis = 0)
    labels = np.delete(labels,del_index,axis = 0)
    return signals,labels


# def sortbylabel(signals,labels):
#     signals


def segment_dataset(signals,labels,a=0.8,random=True):
    length = len(labels)
    if random:
        transformer.shuffledata(signals, labels)
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
def loaddataset(opt,shuffle = False): 
    print('Loading dataset...')
    if opt.separated:
        signals_train = np.load(opt.dataset_dir+'/signals_train.npy')
        labels_train = np.load(opt.dataset_dir+'/labels_train.npy')
        signals_eval = np.load(opt.dataset_dir+'/signals_eval.npy')
        labels_eval = np.load(opt.dataset_dir+'/labels_eval.npy')
        if opt.normliaze != 'None':
            for i in range(signals_train.shape[0]):
                for j in range(signals_train.shape[1]):
                    signals_train[i][j] = arr.normliaze(signals_train[i][j], mode = opt.normliaze, truncated=5)
            for i in range(signals_eval.shape[0]):
                for j in range(signals_eval.shape[1]):
                    signals_eval[i][j] = arr.normliaze(signals_eval[i][j], mode = opt.normliaze, truncated=5)
    else:
        signals = np.load(opt.dataset_dir+'/signals.npy') 
        labels = np.load(opt.dataset_dir+'/labels.npy')
        if opt.normliaze != 'None':
            for i in range(signals.shape[0]):
                for j in range(signals.shape[1]):
                    signals[i][j] = arr.normliaze(signals[i][j], mode = opt.normliaze, truncated=5)

        if not opt.no_shuffle:
            transformer.shuffledata(signals,labels)

    if opt.separated:
        return signals_train,labels_train,signals_eval,labels_eval
    else:
        return signals,labels
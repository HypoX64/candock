import os
import sys
import time
import random
from multiprocessing import Process, Queue

import scipy.io as sio
import numpy as np

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

def rebuild_domain(domain):
    domain = domain.tolist()
    new_domain = np.zeros(len(domain),dtype = np.int64)
    domain_map = {}
    i = 0
    for key in list(set(domain)):
        domain_map[key] = i
        i += 1
    for i in range(len(domain)):
        new_domain[i] = domain_map[domain[i]]
    return np.array(new_domain)  

def preprocess(opt, signals, indexs, queue):
    num,ch = signals.shape[:2]

    for index in indexs:
        signal = signals[index].copy()
        # normliaze
        if opt.normliaze != 'None':
            for i in range(ch):
                signal[i] = arr.normliaze(signal[i], mode = opt.normliaze, truncated=1e2)
        # filter
        if opt.filter != 'None':
            for i in range(ch): 
                if opt.filter == 'fft':
                    signal[i] = dsp.fft_filter(signal[i], opt.filter_fs, opt.filter_fc,type = opt.filter_mod) 
                elif opt.filter == 'iir':         
                    signal[i] = dsp.bpf(signal[i], opt.filter_fs, opt.filter_fc[0], opt.filter_fc[1], numtaps=3, mode='iir')
                elif opt.filter == 'fir':
                    signal[i] = dsp.bpf(signal[i], opt.filter_fs, opt.filter_fc[0], opt.filter_fc[1], numtaps=101, mode='fir')
        
        # wave filter
        if opt.wave != 'None':
            for i in range(ch):
                signal[i] = dsp.wave_filter(signal[i],opt.wave,opt.wave_level,opt.wave_usedcoeffs)
        
        queue.put([signal,index])

#load all data in datasets
def loaddataset(opt): 
    print('Loading dataset...')

    signals = np.load(os.path.join(opt.dataset_dir,'signals.npy'))
    labels = np.load(os.path.join(opt.dataset_dir,'labels.npy'))

    queue = Queue(opt.load_thread)
    pre_thread_num = np.ceil(signals.shape[0]/opt.load_thread).astype(np.int)
    indexs = np.linspace(0,signals.shape[0]-1,num=signals.shape[0],dtype=np.int64)
    for i in range(opt.load_thread):
        p = Process(target=preprocess,args=(opt,signals,indexs[i*pre_thread_num:(i+1)*pre_thread_num],queue))         
        p.daemon = True
        p.start()
    for i in range(signals.shape[0]):
        signal,index = queue.get()
        signals[index] = signal

    if opt.fold_index == 'auto':
        transforms.shuffledata(signals,labels)

    return signals.astype(np.float32),labels.astype(np.int64)
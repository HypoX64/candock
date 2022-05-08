import os
import sys
from multiprocessing import Process, Queue
from scipy import signal
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

sys.path.append("..")
from . import transforms,statistics,augmenter
from util import dsp,util
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
    if opt.use_channel != 'all':
        signals = signals[:,opt.use_channel,:]
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

    return signals.astype(np.float32),labels.astype(np.int64)

class CandockDataset(Dataset):

    def __init__(self,opt,signals,labels,indexs=None,test_flag=False):
        if indexs is not None:
            self.opt,self.singals,self.labels,self.test_flag = opt,signals[indexs],labels[indexs],test_flag
            self.len = len(indexs)
        else:
            self.opt,self.singals,self.labels,self.test_flag = opt,signals,labels,test_flag
            self.len = len(labels)
    
    def __getitem__(self, index):
        signal,label = self.singals[index],np.array(self.labels[index])
        signal = transforms.ToInputShape(self.opt,signal,self.test_flag)
        signal,label = transforms.ToTensor(signal),transforms.ToTensor(label)
        return signal,label

    def __len__(self):
        return self.len

class CandockDomainDataset(Dataset):

    def __init__(self,opt,signals,labels,indexs,test_flag=False):
        self.opt,self.singals,self.labels,self.test_flag = opt,signals[indexs],labels[indexs],test_flag
        self.len = len(indexs)
    
    def __getitem__(self, index):
        signal,label = self.singals[index],self.labels[index]
        signal = transforms.ToInputShape(self.opt,signal,self.test_flag)
        signal,label = transforms.ToTensor(signal),transforms.ToTensor(label)
        return signal,label

    def __len__(self):
        return self.len

def GetLoader(opt,dataset):
    return DataLoader(  dataset=dataset,
                        batch_size=opt.batchsize,
                        shuffle=True,
                        num_workers= opt.load_thread
                    )
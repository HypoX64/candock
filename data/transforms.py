import os
import random
import numpy as np
import torch

import sys
sys.path.append("..")
from util import dsp
from util import array_operation as arr
from . import augmenter

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

def k_fold_generator(length,fold_num,fold_index = 'auto'):
    sequence = np.linspace(0,length-1,length,dtype='int')
    train_sequence = [];eval_sequence = []
    if fold_num == 0 or fold_num == 1:
        if fold_index != 'auto' :
            fold_index = [0]+fold_index+[length]
        else:
            fold_index = [0]+[int(length*0.8)]+[length]
        train_sequence.append(sequence[:fold_index[1]])
        eval_sequence.append(sequence[fold_index[1]:])
    else:
        if fold_index != 'auto' :
            fold_index = [0]+fold_index+[length]
        else:
            fold_index = []
            for i in range(fold_num):
                fold_index.append(length//fold_num*i)
            fold_index.append(length)
        for i in range(len(fold_index)-1):
            eval_sequence.append(sequence[fold_index[i]:fold_index[i+1]])
            train_sequence.append(np.concatenate((sequence[0:fold_index[i]],sequence[fold_index[i+1]:]),axis=0))
    if fold_num > 1:
        print('fold_index:',fold_index)
    return train_sequence,eval_sequence


def batch_generator(data,target,sequence,shuffle = True):
    batchsize = len(sequence)
    out_data = np.zeros((batchsize,data.shape[1],data.shape[2]), data.dtype)
    out_target = np.zeros((batchsize), target.dtype)
    for i in range(batchsize):
        out_data[i] = data[sequence[i]]
        out_target[i] = target[sequence[i]]
    return out_data,out_target


def ToTensor(data,target=None,gpu_id=0):
    if target is not None:
        data = torch.from_numpy(data)
        target = torch.from_numpy(target)
        if gpu_id != -1:
            data = data.cuda()
            target = target.cuda()
        return data,target
    else:
        data = torch.from_numpy(data)
        if gpu_id != -1:
            data = data.cuda()
        return data

def ToInputShape(opt,data,test_flag = False):

    if opt.model_type == '1d':
        result = augmenter.base1d(opt, data, test_flag = test_flag).astype(np.float32)

    elif opt.model_type == '2d':
        _batchsize,_ch,_size = data.shape
        h,w = opt.stft_shape
        data = augmenter.base1d(opt, data, test_flag = test_flag)
        result = np.zeros((_batchsize,_ch,h,w), dtype=np.float32)
        for i in range(_batchsize):
            for j in range(opt.input_nc):
                result[i][j] = dsp.signal2spectrum(data[i][j],opt.stft_size,opt.stft_stride,
                    opt.cwt_wavename,opt.cwt_scale_num,opt.spectrum_n_downsample,not opt.stft_no_log, mod=opt.spectrum)

    return result

import os
import random
import numpy as np
import torch
from . import dsp
from . import array_operation as arr

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)
    # return data,target

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
        data = torch.from_numpy(data).float()
        target = torch.from_numpy(target).long()
        if gpu_id != -1:
            data = data.cuda()
            target = target.cuda()
        return data,target
    else:
        data = torch.from_numpy(data).float()
        if gpu_id != -1:
            data = data.cuda()
        return data

def random_transform_1d(data,opt,test_flag):
    batchsize,ch,length = data.shape

    if test_flag:
        move = int((length-opt.finesize)*0.5)
        result = data[:,:,move:move+opt.finesize]
    else:
        #random scale
        if 'scale' in opt.augment:
            length = np.random.randint(opt.finesize, length*1.1, dtype=np.int64)
            result = np.zeros((batchsize,ch,length))
            for i in range(batchsize):
                for j in range(ch):
                    result[i][j] = arr.interp(data[i][j], length)
            data = result

        #random crop    
        move = int((length-opt.finesize)*random.random())
        result = data[:,:,move:move+opt.finesize]

        #random flip
        if 'flip' in opt.augment:
            if random.random()<0.5:
                result = result[:,:,::-1]
        
        #random amp
        if 'amp' in opt.augment:
            result = result*random.uniform(0.9,1.1)

        #add noise
        if 'noise' in opt.augment:
            noise = np.random.rand(ch,opt.finesize)
            result = result + (noise-0.5)*0.01

    return result

def random_transform_2d(img,finesize = (224,244),test_flag = True):
    h,w = img.shape[:2]
    if test_flag:
        h_move = int((h-finesize[0])*0.5)
        w_move = int((w-finesize[1])*0.5)
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
    else:
        #random crop
        h_move = int((h-finesize[0])*random.random())
        w_move = int((w-finesize[1])*random.random())
        result = img[h_move:h_move+finesize[0],w_move:w_move+finesize[1]]
        #random flip
        if random.random()<0.5:
            result = result[:,::-1]
        #random amp
        result = result*random.uniform(0.9,1.1)+random.uniform(-0.05,0.05)
    return result

def ToInputShape(data,opt,test_flag = False):
    #data = data.astype(np.float32)
    _batchsize,_ch,_size = data.shape

    if opt.model_type == '1d':
        result = random_transform_1d(data, opt, test_flag = test_flag)

    elif opt.model_type == '2d':
        result = []
        h,w = opt.stft_shape
        for i in range(_batchsize):
            for j in range(opt.input_nc):
                spectrum = dsp.signal2spectrum(data[i][j],opt.stft_size,opt.stft_stride, opt.stft_n_downsample, not opt.stft_no_log)
                spectrum = random_transform_2d(spectrum,(h,int(w*0.9)),test_flag=test_flag)
                result.append(spectrum)
        result = (np.array(result)).reshape(_batchsize,opt.input_nc,h,int(w*0.9))

    return result

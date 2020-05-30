import os
import random
import numpy as np
import torch
from . import dsp
from . import array_operation as arr
# import dsp

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)
    # return data,target

def k_fold_generator(length,fold_num,separated=False):
    if separated:
        sequence = np.linspace(0, length-1,num = length,dtype='int')
        return sequence
    else:
        if fold_num == 0 or fold_num == 1:
            train_sequence = np.linspace(0,int(length*0.8)-1,int(length*0.8),dtype='int')[None]
            test_sequence = np.linspace(int(length*0.8),length-1,int(length*0.2),dtype='int')[None]
        else:
            sequence = np.linspace(0,length-1,length,dtype='int')
            train_length = int(length/fold_num*(fold_num-1))
            test_length = int(length/fold_num)
            train_sequence = np.zeros((fold_num,train_length), dtype = 'int')
            test_sequence = np.zeros((fold_num,test_length), dtype = 'int')
            for i in range(fold_num):
                test_sequence[i] = (sequence[test_length*i:test_length*(i+1)])[:test_length]
                train_sequence[i] = np.concatenate((sequence[0:test_length*i],sequence[test_length*(i+1):]),axis=0)[:train_length]
        return train_sequence,test_sequence

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

def random_transform_1d(data,finesize,test_flag):
    batch_size,ch,length = data.shape

    if test_flag:
        move = int((length-finesize)*0.5)
        result = data[:,:,move:move+finesize]
    else:
        #random crop    
        move = int((length-finesize)*random.random())
        result = data[:,:,move:move+finesize]
        #random flip
        if random.random()<0.5:
            result = result[:,:,::-1]
        #random amp
        result = result*random.uniform(0.9,1.1)
        #add noise
        # noise = np.random.rand(ch,finesize)
        # result = result + (noise-0.5)*0.01
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

    if opt.model_type == '1d':
        result = random_transform_1d(data, opt.finesize, test_flag=test_flag)

    elif opt.model_type == '2d':
        result = []
        h,w = opt.stft_shape
        for i in range(opt.batchsize):
            for j in range(opt.input_nc):
                spectrum = dsp.signal2spectrum(data[i][j],opt.stft_size,opt.stft_stride, opt.stft_n_downsample, not opt.stft_no_log)
                spectrum = random_transform_2d(spectrum,(h,int(w*0.9)),test_flag=test_flag)
                result.append(spectrum)
        result = (np.array(result)).reshape(opt.batchsize,opt.input_nc,h,int(w*0.9))

    return result

import numpy as np
import os
import torch
import random
import DSP

def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)
    # return data,target


def batch_generator(data,target,batchsize,shuffle = True):
    if shuffle:
        shuffledata(data,target)

    data = trimdata(data,batchsize)
    target = trimdata(target,batchsize)
    data = data.reshape(-1,batchsize,3000)
    target = target.reshape(-1,batchsize)
    return data[0:int(0.8*len(target))],target[0:int(0.8*len(target))],data[int(0.8*len(target)):],target[int(0.8*len(target)):]
    
def Normalize(data,maxmin,avg,sigma):
    data = np.clip(data, -maxmin, maxmin)
    return (data-avg)/sigma

def ToTensor(data,target,no_cuda = False):

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target).long()
    if not no_cuda:
        data = data.cuda()
        target = target.cuda()
    return data,target

def random_transform_1d(data,finesize,test_flag):
    length = len(data)
    if test_flag:
        move = int((length-finesize)*0.5)
        result = data[move:move+finesize]
    else:
        #random crop    
        move = int((length-finesize)*random.random())
        result = data[move:move+finesize]

        #random flip
        if random.random()<0.5:
            result = result[::-1]

        #random amp
        result = result*random.uniform(0.95,1.05)

    return result

def random_transform_2d(img,finesize,test_flag):
    h,w = img.shape[:2]
    if test_flag:
        h_move = 1
        w_move = int((w-finesize)*0.5)
        result = img[h_move:h_move+finesize,w_move:w_move+finesize]
    else:
        #random crop
        h_move = int(3*random.random()) #do not loss low freq signal infos
        w_move = int((w-finesize)*random.random())
        result = img[h_move:h_move+finesize,w_move:w_move+finesize]
        #random flip
        if random.random()<0.5:
            result = result[:,::-1]
        #random amp
        result = result*random.uniform(0.98,1.02)+random.uniform(-0.01,0.01)
    return result
    

def ToInputShape(data,net_name,norm=True,test_flag = False):
    data = data.astype(np.float32)
    batchsize=data.shape[0]
    if net_name=='LSTM':
        result =[]
        for i in range(0,batchsize):
            randomdata=random_transform_1d(data[i],finesize = 2700,test_flag=test_flag)
            result.append(DSP.getfeature(randomdata))
        result = np.array(result).reshape(batchsize,2700*5)
    elif net_name=='CNN' or net_name=='resnet18_1d':
        result =[]
        for i in range(0,batchsize):
            randomdata=random_transform_1d(data[i],finesize = 2700,test_flag=test_flag)
            # result.append(DSP.getfeature(randomdata,ch_num = 6))
            result.append(randomdata)
        result = np.array(result)
        if norm:
            result = Normalize(result,maxmin = 200,avg=0,sigma=200)
        result = result.reshape(batchsize,1,2700)

    elif net_name in ['resnet18','densenet121','densenet201','resnet101','resnet50']:
        result =[]
        for i in range(0,batchsize):
            spectrum = DSP.signal2spectrum(data[i])
            spectrum = random_transform_2d(spectrum,224,test_flag=test_flag)
            result.append(spectrum)
        result = np.array(result)
        if norm:
            #std,mean,median,max= 0.2972 0.3008 0.2006 2.0830
            result=Normalize(result,2,0.3,1)
        result = result.reshape(batchsize,1,224,224)
        # print(result.shape)

    return result


# datasetpath='/media/hypo/Hypo/training'
# dir = '/media/hypo/Hypo/training/tr03-0005'

def main():
    dir = '/media/hypo/Hypo/physionet_org_train/tr03-0052'
    t1=time.time()
    stages=loadstages(dir)
    for i in range(len(stages)):
        if stages[i]!=5:
            print(i+1)
            break
    print(stages.shape)

    t2=time.time()
    print(t2-t1)

if __name__ == '__main__':
    main()

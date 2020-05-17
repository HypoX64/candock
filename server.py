import os
import time
import shutil
import numpy as np
import random
import torch
from torch import nn, optim
import warnings

from util import util,transformer,dataloader,statistics,plot,options
from util import array_operation as arr
from models import creatnet,core

opt = options.Options()
opt.parser.add_argument('--ip',type=str,default='', help='')
opt = opt.getparse()
torch.cuda.set_device(opt.gpu_id)
opt.k_fold = 0
opt.save_dir = './datasets/server/tmp'
util.makedirs(opt.save_dir)
'''load ori data'''
signals,labels = dataloader.loaddataset(opt)
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
opt = options.get_auto_options(opt, label_cnt_per, label_num, signals.shape)

'''def network'''
core = core.Core(opt)
core.network_init(printflag=True)

'''Receive data'''
if os.path.isdir('./datasets/server/data'):
    shutil.rmtree('./datasets/server/data')
os.system('unzip ./datasets/server/data.zip -d ./datasets/server/')
categorys = os.listdir('./datasets/server/data')
categorys.sort()
print('categorys:',categorys)
receive_category = len(categorys)
received_signals = []
received_labels = []
for i in range(receive_category):
    samples = os.listdir(os.path.join('./datasets/server/data',categorys[i]))

    for sample in samples:
        txt = util.loadtxt(os.path.join('./datasets/server/data',categorys[i],sample))
        #print(os.path.join('./datasets/server/data',categorys[i],sample))
        txt_split = txt.split()
        signal_ori = np.zeros(len(txt_split))
        for point in range(len(txt_split)):
            signal_ori[point] = float(txt_split[point])
        # #just cut
        # for j in range(1,len(signal_ori)//opt.loadsize-1):
        #     this_signal = signal_ori[j*opt.loadsize:(j+1)*opt.loadsize]
        #     this_signal = arr.normliaze(this_signal,'5_95',truncated=4)
        #     received_signals.append(this_signal)
        #     received_labels.append(i)
        #random cut
        for j in range(500//len(samples)-1):
            ran = random.randint(1000, len(signal_ori)-2000-1)
            this_signal = signal_ori[ran:ran+2000]
            this_signal = arr.normliaze(this_signal,'5_95',truncated=4)
            received_signals.append(this_signal)
            received_labels.append(i)

received_signals = np.array(received_signals).reshape(-1,opt.input_nc,opt.loadsize)
received_labels = np.array(received_labels).reshape(-1,1)

# print(labels)
'''merge data'''
signals = signals[receive_category*500:]
labels = labels[receive_category*500:]
signals = np.concatenate((signals, received_signals))
labels = np.concatenate((labels, received_labels))
transformer.shuffledata(signals,labels)


label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
opt = options.get_auto_options(opt, label_cnt_per, label_num, signals.shape)
train_sequences,test_sequences = transformer.k_fold_generator(len(labels),opt.k_fold)

for epoch in range(opt.epochs):
    t1 = time.time()
    core.train(signals,labels,train_sequences[0])
    core.eval(signals,labels,test_sequences[0])
    t2=time.time()
    if epoch+1==1:
        util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)
core.save_traced_net()

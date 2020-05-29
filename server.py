import os
import time
import shutil
import numpy as np
import random
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
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
# use separated mode
signals_train,labels_train,signals_eval,labels_eval = dataloader.loaddataset(opt)
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels_train)
opt = options.get_auto_options(opt, label_cnt_per, label_num, signals_train.shape)
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
category_num = len(categorys)
# received_signals_train = [];received_labels_train = []
# received_signals_eval = [];received_labels_eval = []

# sample_num = 1000
# eval_num = 1
# for i in range(category_num):
#     samples = os.listdir(os.path.join('./datasets/server/data',categorys[i]))

#     for j in range(len(samples)):
#         txt = util.loadtxt(os.path.join('./datasets/server/data',categorys[i],samples[j]))
#         #print(os.path.join('./datasets/server/data',categorys[i],sample))
#         txt_split = txt.split()
#         signal_ori = np.zeros(len(txt_split))
#         for point in range(len(txt_split)):
#             signal_ori[point] = float(txt_split[point])

#         for x in range(sample_num//len(samples)):
#             ran = random.randint(1000, len(signal_ori)-2000-1)
#             this_signal = signal_ori[ran:ran+2000]
#             this_signal = arr.normliaze(this_signal,'5_95',truncated=4)
#             # if i ==0:
#             #     plt.plot(this_signal)
#             #     plt.show()
#             if j < (len(samples)-eval_num):               
#                 received_signals_train.append(this_signal)
#                 received_labels_train.append(i)
#             else:
#                 received_signals_eval.append(this_signal)
#                 received_labels_eval.append(i)

# received_signals_train = np.array(received_signals_train).reshape(-1,opt.input_nc,opt.loadsize)
# received_labels_train = np.array(received_labels_train).reshape(-1,1)
# received_signals_eval = np.array(received_signals_eval).reshape(-1,opt.input_nc,opt.loadsize)
# received_labels_eval = np.array(received_labels_eval).reshape(-1,1)
#print(received_signals_train.shape,received_signals_eval.shape)

received_signals = [];received_labels = []

sample_num = 1000
eval_num = 1
for i in range(category_num):
    samples = os.listdir(os.path.join('./datasets/server/data',categorys[i]))
    random.shuffle(samples)
    for j in range(len(samples)):
        txt = util.loadtxt(os.path.join('./datasets/server/data',categorys[i],samples[j]))
        #print(os.path.join('./datasets/server/data',categorys[i],sample))
        txt_split = txt.split()
        signal_ori = np.zeros(len(txt_split))
        for point in range(len(txt_split)):
            signal_ori[point] = float(txt_split[point])

        for x in range(sample_num//len(samples)):
            ran = random.randint(1000, len(signal_ori)-2000-1)
            this_signal = signal_ori[ran:ran+2000]
            this_signal = arr.normliaze(this_signal,'5_95',truncated=4)
            
            received_signals.append(this_signal)
            received_labels.append(i)

received_signals = np.array(received_signals).reshape(-1,opt.input_nc,opt.loadsize)
received_labels = np.array(received_labels).reshape(-1,1)
received_signals_train,received_labels_train,received_signals_eval,received_labels_eval=\
dataloader.segment_dataset(received_signals, received_labels, 0.8,random=False)

print(received_signals_train.shape,received_signals_eval.shape)

# print(labels)
'''merge data'''
signals_train,labels_train = dataloader.del_labels(signals_train,labels_train, np.linspace(0, category_num-1,category_num,dtype=np.int64))
signals_eval,labels_eval = dataloader.del_labels(signals_eval,labels_eval, np.linspace(0, category_num-1,category_num,dtype=np.int64))


signals_train = np.concatenate((signals_train, received_signals_train))
labels_train = np.concatenate((labels_train, received_labels_train))

signals_eval = np.concatenate((signals_eval, received_signals_eval))
labels_eval = np.concatenate((labels_eval, received_labels_eval))


label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels_train)
opt = options.get_auto_options(opt, label_cnt_per, label_num, signals_train.shape)
train_sequences= transformer.k_fold_generator(len(labels_train),opt.k_fold,opt.separated)
eval_sequences= transformer.k_fold_generator(len(labels_eval),opt.k_fold,opt.separated)


for epoch in range(opt.epochs):
    t1 = time.time()
    if opt.separated:
        #print(signals_train.shape,labels_train.shape)
        core.train(signals_train,labels_train,train_sequences)
        core.eval(signals_eval,labels_eval,eval_sequences)
    else:            
        core.train(signals,labels,train_sequences[fold])
        core.eval(signals,labels,eval_sequences[fold])
    t2=time.time()
    if epoch+1==1:
        util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)
plot.draw_heatmap(core.confusion_mats[-1],opt,name = 'final')
core.save_traced_net()

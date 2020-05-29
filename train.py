import os
import time

import numpy as np
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

from util import util,transformer,dataloader,statistics,plot,options
from models import core

opt = options.Options().getparse()
t1 = time.time()

'''
Use your own data to train
* step1: Generate signals.npy and labels.npy in the following format.
# 1.type:numpydata   signals:np.float64   labels:np.int64
# 2.shape  signals:[num,ch,length]    labels:[num]
# num:samples_num, ch :channel_num,  num:length of each sample
# for example:
signals = np.zeros((10,1,10),dtype='np.float64')
labels = np.array([0,0,0,0,0,1,1,1,1,1])      #0->class0    1->class1
* step2: input  ```--dataset_dir your_dataset_dir``` when running code.
'''

#----------------------------Load Data----------------------------
if opt.separated:
    signals_train,labels_train,signals_eval,labels_eval = dataloader.loaddataset(opt)
    label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels_train)
    util.writelog('label statistics: '+str(label_cnt),opt,True)
    opt = options.get_auto_options(opt, label_cnt_per, label_num, signals_train.shape)
    train_sequences= transformer.k_fold_generator(len(labels_train),opt.k_fold,opt.separated)
    eval_sequences= transformer.k_fold_generator(len(labels_eval),opt.k_fold,opt.separated)
else:
    signals,labels = dataloader.loaddataset(opt)
    label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
    util.writelog('label statistics: '+str(label_cnt),opt,True)
    opt = options.get_auto_options(opt, label_cnt_per, label_num, signals.shape)
    train_sequences,eval_sequences = transformer.k_fold_generator(len(labels),opt.k_fold)
t2 = time.time()
print('load data cost time: %.2f'% (t2-t1),'s')

core = core.Core(opt)
core.network_init(printflag=True)

print('begin to train ...')
fold_final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
for fold in range(opt.k_fold):
    if opt.k_fold != 1:util.writelog('------------------------------ k-fold:'+str(fold+1)+' ------------------------------',opt,True)
    core.network_init()
    final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    for epoch in range(opt.epochs):  
        t1 = time.time()
        
        if opt.separated:
            #print(signals_train.shape,labels_train.shape)
            core.train(signals_train,labels_train,train_sequences)
            core.eval(signals_eval,labels_eval,eval_sequences)
        else:            
            core.train(signals,labels,train_sequences[fold])
            core.eval(signals,labels,eval_sequences[fold])
        core.save()

        t2=time.time()
        if epoch+1==1:
            util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)
    
    #save result
    if opt.model_name != 'autoencoder':
        pos = core.plot_result['eval'].index(min(core.plot_result['eval']))-1
        final_confusion_mat = core.confusion_mats[pos]
        if opt.k_fold==1:
            statistics.statistics(final_confusion_mat, opt, 'final', 'final_eval')
            np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), final_confusion_mat)
        else:
            fold_final_confusion_mat += final_confusion_mat
            util.writelog('fold  -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(final_confusion_mat)),opt,True)
            util.writelog('confusion_mat:\n'+str(final_confusion_mat)+'\n',opt,True)
            plot.draw_heatmap(final_confusion_mat,opt,name = 'fold'+str(fold+1)+'_eval')

if opt.model_name != 'autoencoder':
    if opt.k_fold != 1:
        statistics.statistics(fold_final_confusion_mat, opt, 'final', 'k-fold-final_eval')
        np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), fold_final_confusion_mat)
        
    if opt.mergelabel:
        mat = statistics.mergemat(fold_final_confusion_mat, opt.mergelabel)
        statistics.statistics(mat, opt, 'merge', 'mergelabel_final')

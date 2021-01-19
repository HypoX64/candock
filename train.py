import os
import time

import numpy as np
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
import core

opt = options.Options().getparse()

"""Use your own data to train
* step1: Generate signals.npy and labels.npy in the following format.
# 1.type:numpydata   signals:np.float32   labels:np.int64
# 2.shape  signals:[num,ch,length]    labels:[num]
# num:samples_num, ch :channel_num,  length:length of each sample
# for example:
signals = np.zeros((10,1,10),dtype='np.float64')
labels = np.array([0,0,0,0,0,1,1,1,1,1])      #0->class0    1->class1
* step2: input  ```--dataset_dir your_dataset_dir``` when running code.
"""

#----------------------------Load Data----------------------------
t1 = time.time()
signals,labels = dataloader.loaddataset(opt)
if opt.gan:
    signals,labels = augmenter.dcgan(opt,signals,labels)
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
util.writelog('label statistics: '+str(label_cnt),opt,True)
opt = options.get_auto_options(opt, signals, labels)
train_sequences,eval_sequences = transforms.k_fold_generator(len(labels),opt.k_fold,opt.fold_index)
t2 = time.time()
print('Cost time: %.2f'% (t2-t1),'s')

core = core.Core(opt)
core.network_init(printflag=True)

print('Begin to train ...')
final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
final_results = {}
for fold in range(opt.k_fold):
    if opt.k_fold != 1:util.writelog('------------------------------ k-fold:'+str(fold+1)+' ------------------------------',opt,True)
    core.fold = fold
    core.network_init()

    for epoch in range(opt.epochs): 

        if opt.mode in ['classify_1d','classify_2d','autoencoder']: 
            core.train(signals,labels,train_sequences[fold])
        elif opt.model_name in ['dann','dann_base']:
            core.dann_train(signals,labels,train_sequences[fold],eval_sequences[fold])
            
        core.eval(signals,labels,eval_sequences[fold])
        core.epoch_save()
        core.check_remain_time()

    final_results[fold] = core.results

    #save result
    if opt.mode != 'autoencoder':
        fold_best_confusion_mat = core.results['confusion_mat'][core.results['best_epoch']]
        final_confusion_mat += fold_best_confusion_mat
        if opt.k_fold != 1:
            util.writelog('fold'+str(fold+1)+' -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(fold_best_confusion_mat)),opt,True,True)
            util.writelog('confusion_mat:\n'+str(fold_best_confusion_mat)+'\n',opt,True,False)

if opt.mode != 'autoencoder':
    statistics.statistics(final_confusion_mat, opt, 'final', 'final_eval')
    np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), final_confusion_mat)
    statistics.save_detail_results(opt, final_results)
        
    if opt.mergelabel:
        mat = statistics.mergemat(final_confusion_mat, opt.mergelabel)
        statistics.statistics(mat, opt, 'merge', 'mergelabel_final')

plot.final(opt, final_results)
util.copyfile(opt.tensorboard, os.path.join(opt.save_dir,'runs',os.path.split(opt.tensorboard)[1]))
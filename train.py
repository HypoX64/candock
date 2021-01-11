import os
import time

import numpy as np
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
from models import core

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
fold_final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
eval_detail = [[],[],[]]
for fold in range(opt.k_fold):
    if opt.k_fold != 1:util.writelog('------------------------------ k-fold:'+str(fold+1)+' ------------------------------',opt,True)
    core.fold = fold
    core.network_init()
    final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    for epoch in range(opt.epochs): 

        if opt.mode in ['classify_1d','classify_2d','autoencoder']: 
            core.train(signals,labels,train_sequences[fold])
        elif opt.model_name in ['dann','dann_base']:
            core.dann_train(signals,labels,train_sequences[fold],eval_sequences[fold])
            
        core.eval(signals,labels,eval_sequences[fold])
        core.epoch_save()
        core.check_remain_time()

        if opt.eval_detail:
            for i in range(3):eval_detail[i] += core.eval_detail[i]
    #save result
    if opt.mode != 'autoencoder':
        if opt.best_index =='f1':
            pos = core.results['F1'].index(max(core.results['F1']))
        elif opt.best_index =='err':
            pos = core.results['err'].index(min(core.results['err']))
        final_confusion_mat = core.confusion_mats[pos]
        if opt.k_fold==1:
            statistics.statistics(final_confusion_mat, opt, 'final', 'final_eval')
            np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), final_confusion_mat)
        else:
            fold_final_confusion_mat += final_confusion_mat
            util.writelog('fold  -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(final_confusion_mat)),opt,True,False)
            util.writelog('confusion_mat:\n'+str(final_confusion_mat)+'\n',opt,True)
            # plot.draw_heatmap(final_confusion_mat,opt,name = 'fold'+str(fold+1)+'_eval',step=fold)

if opt.eval_detail:
    statistics.eval_detail(opt,eval_detail)

if opt.mode != 'autoencoder':
    if opt.k_fold != 1:
        statistics.statistics(fold_final_confusion_mat, opt, 'final', 'k-fold-final_eval')
        np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), fold_final_confusion_mat)
        
    if opt.mergelabel:
        mat = statistics.mergemat(fold_final_confusion_mat, opt.mergelabel)
        statistics.statistics(mat, opt, 'merge', 'mergelabel_final')

util.copyfile(opt.tensorboard, os.path.join(opt.save_dir,'runs',os.path.split(opt.tensorboard)[1]))
import os
import sys
import time

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
from models import model_util
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
    if opt.mode in ['classify_1d','classify_2d','dml']: 
        train_dataset = dataloader.CandockDataset(opt,signals,labels,train_sequences[fold])
        eval_dataset  = dataloader.CandockDataset(opt,signals,labels,eval_sequences[fold],test_flag=True)
        train_loader = dataloader.GetLoader(opt,train_dataset)
        eval_loader = dataloader.GetLoader(opt,eval_dataset)

    elif opt.mode in ['domain','domain_1d']:
        pass

    for epoch in range(opt.n_epochs): 
        if opt.mode in ['classify_1d','classify_2d']: 
            core.train(train_loader)
            core.eval(eval_loader)
        elif opt.mode == 'dml':
            core.dml_train(train_loader)
            core.dml_eval(train_dataset,eval_dataset)

        elif opt.mode in ['domain','domain_1d']:
            pass
            # core.dann_train(signals,labels,train_sequences[fold],eval_sequences[fold])
            # core.dann_eval(signals,labels,eval_sequences[fold])
            
        core.epoch_save()
        core.check_remain_time()

    #save result
    if opt.mode != 'dml':
        final_results[fold] = core.results
        fold_best_confusion_mat = core.results['confusion_mat'][core.results['best_epoch']]
        final_confusion_mat += fold_best_confusion_mat
        if opt.k_fold != 1:
            util.writelog('fold'+str(fold+1)+' -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(fold_best_confusion_mat)),opt,True,True)
            util.writelog('confusion_mat:\n'+str(fold_best_confusion_mat)+'\n',opt,True,False)

if opt.mode != 'dml':
    statistics.statistics(final_confusion_mat, opt, 'final', 'final_eval')
    np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), final_confusion_mat)
    statistics.save_detail_results(opt, final_results)
        
    if opt.mergelabel != 'None':
        mat = statistics.mergemat(final_confusion_mat, opt.mergelabel)
        statistics.statistics(mat, opt, 'merge', 'mergelabel_final')

    plot.final(opt, final_results)
util.copyfile(opt.tensorboard_save_dir, os.path.join(opt.save_dir,'runs',os.path.split(opt.tensorboard_save_dir)[1]))
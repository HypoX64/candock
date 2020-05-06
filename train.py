import os
import time

import numpy as np
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

from util import util,transformer,dataloader,statistics,heatmap,options
from models import creatnet

opt = options.Options().getparse()
torch.cuda.set_device(opt.gpu_id)
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

signals,labels = dataloader.loaddataset(opt)
label_cnt,label_cnt_per,_ = statistics.label_statistics(labels)
train_sequences,test_sequences = transformer.k_fold_generator(len(labels),opt.k_fold)
t2 = time.time()
print('load data cost time: %.2f'% (t2-t1),'s')

net=creatnet.CreatNet(opt)
util.writelog('network:\n'+str(net),opt,True)

util.show_paramsnumber(net,opt)
weight = np.ones(opt.label)
if opt.weight_mod == 'auto':
    weight = 1/label_cnt_per
    weight = weight/np.min(weight)
util.writelog('label statistics: '+str(label_cnt),opt,True)
util.writelog('Loss_weight:'+str(weight),opt,True)
weight = torch.from_numpy(weight).float()
# print(net)

if not opt.no_cuda:
    net.cuda()
    weight = weight.cuda()
if not opt.no_cudnn:
    torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
criterion = nn.CrossEntropyLoss(weight)
torch.save(net.cpu().state_dict(),os.path.join(opt.save_dir,'tmp.pth'))

def evalnet(net,signals,labels,sequences,epoch,plot_result={}):
    # net.eval()
    confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    for i in range(int(len(sequences)/opt.batchsize)):
        signal,label = transformer.batch_generator(signals, labels, sequences[i*opt.batchsize:(i+1)*opt.batchsize])
        signal = transformer.ToInputShape(signal,opt,test_flag =True)
        signal,label = transformer.ToTensor(signal,label,no_cuda =opt.no_cuda)
        with torch.no_grad():
            out = net(signal)
        pred = torch.max(out, 1)[1]

        pred=pred.data.cpu().numpy()
        label=label.data.cpu().numpy()
        for x in range(len(pred)):
            confusion_mat[label[x]][pred[x]] += 1

    recall,acc,sp,err,k  = statistics.report(confusion_mat)
    plot_result['test'].append(err)   
    heatmap.draw(confusion_mat,opt,name = 'current_test')
    print('epoch:'+str(epoch),' macro-prec,reca,F1,err,kappa: '+str(statistics.report(confusion_mat)))
    return plot_result,confusion_mat

print('begin to train ...')
fold_final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
for fold in range(opt.k_fold):
    if opt.k_fold != 1:util.writelog('------------------------------ k-fold:'+str(fold+1)+' ------------------------------',opt,True)
    iter_cnt = 0
    net.load_state_dict(torch.load(os.path.join(opt.save_dir,'tmp.pth')))
    if opt.pretrained:
        net.load_state_dict(torch.load(os.path.join(opt.save_dir,'pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth')))
    if opt.continue_train:
        net.load_state_dict(torch.load(os.path.join(opt.save_dir,'last.pth')))
    if not opt.no_cuda:
        net.cuda()

    final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    confusion_mats = []
    plot_result={'train':[1.],'test':[1.]}
    
    for epoch in range(opt.epochs):
        t1 = time.time()
        np.random.shuffle(train_sequences[fold])
        net.train()
        for i in range(int(len(train_sequences[fold])/opt.batchsize)):
            signal,label = transformer.batch_generator(signals, labels, train_sequences[fold][i*opt.batchsize:(i+1)*opt.batchsize])
            signal = transformer.ToInputShape(signal,opt,test_flag =False)
            signal,label = transformer.ToTensor(signal,label,no_cuda =opt.no_cuda)

            out = net(signal)
            loss = criterion(out, label)
            pred = torch.max(out, 1)[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred=pred.data.cpu().numpy()
            label=label.data.cpu().numpy()
            for x in range(len(pred)):
                confusion_mat[label[x]][pred[x]] += 1
            iter_cnt += 1
            if iter_cnt%opt.plotfreq==0:       
                plot_result['train'].append(statistics.report(confusion_mat)[3])
                heatmap.draw(confusion_mat,opt,name = 'current_train')
                statistics.plotloss(plot_result,epoch+i/(train_sequences.shape[1]/opt.batchsize),opt)
                confusion_mat[:]=0

        plot_result,confusion_mat_eval = evalnet(net,signals,labels,test_sequences[fold],epoch+1,plot_result)
        confusion_mats.append(confusion_mat_eval)

        torch.save(net.cpu().state_dict(),os.path.join(opt.save_dir,'last.pth'))
        if (epoch+1)%opt.network_save_freq == 0:
            torch.save(net.cpu().state_dict(),os.path.join(opt.save_dir,opt.model_name+'_epoch'+str(epoch+1)+'.pth'))
            print('network saved.')
        if not opt.no_cuda:
            net.cuda()

        t2=time.time()
        if epoch+1==1:
            util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)

    #save result
    pos = plot_result['test'].index(min(plot_result['test']))-1
    final_confusion_mat = confusion_mats[pos]
    if opt.k_fold==1:
        statistics.statistics(final_confusion_mat, opt, 'final', 'final_test')
        np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), final_confusion_mat)
    else:
        fold_final_confusion_mat += final_confusion_mat
        util.writelog('fold  -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(final_confusion_mat)),opt,True)
        util.writelog('confusion_mat:\n'+str(final_confusion_mat)+'\n',opt,True)
        heatmap.draw(final_confusion_mat,opt,name = 'fold'+str(fold+1)+'_test')

if opt.k_fold != 1:
    statistics.statistics(fold_final_confusion_mat, opt, 'final', 'k-fold-final_test')
    np.save(os.path.join(opt.save_dir,'confusion_mat.npy'), fold_final_confusion_mat)
    
if opt.mergelabel:
    mat = statistics.mergemat(fold_final_confusion_mat, opt.mergelabel)
    statistics.statistics(mat, opt, 'merge', 'mergelabel_final')

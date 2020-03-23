import os
import time

import numpy as np
import torch
from torch import nn, optim
import warnings
warnings.filterwarnings("ignore")

import util
import transformer
import dataloader
import statistics
import heatmap
from creatnet import CreatNet
from options import Options

'''
change your own data to train
but the data needs meet the following conditions: 
1.type   numpydata  signals:np.float16  labels:np.int16
2.shape             signals:[num,ch,length]   labels:[num]
3.input signal data should be normalized!!
  we recommend signal data normalized useing 5_95_th for each subject, 
  example: signals_normalized=transformer.Balance_individualized_differences(signals_origin, '5_95_th')
'''

opt = Options().getparse()
torch.cuda.set_device(opt.gpu_id)
t1 = time.time()

signals,labels = dataloader.loaddataset(opt)
label_cnt,label_cnt_per,_ = statistics.label_statistics(labels)
signals,labels = transformer.batch_generator(signals,labels,opt.batchsize,shuffle = False)
train_sequences,test_sequences = transformer.k_fold_generator(len(labels),opt.k_fold)
show_freq = int(len(train_sequences[0])/5)


t2 = time.time()
print('load data cost time: %.2f'% (t2-t1),'s')

net=CreatNet(opt)
util.writelog('network:\n'+str(net),opt,True)

util.show_paramsnumber(net,opt)
weight = np.ones(opt.label)
if opt.weight_mod == 'auto':
    weight = np.log(1/label_cnt_per)
    weight = weight/np.median(weight)
    weight = np.clip(weight, 0.8, 2)
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
    for i, sequence in enumerate(sequences, 1):

        signal=transformer.ToInputShape(signals[sequence],opt,test_flag =True)
        signal,label = transformer.ToTensor(signal,labels[sequence],no_cuda =opt.no_cuda)
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

    net.load_state_dict(torch.load(os.path.join(opt.save_dir,'tmp.pth')))
    if opt.pretrained:
        net.load_state_dict(torch.load(os.path.join(opt.save_dir,'pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth')))
    if opt.continue_train:
        net.load_state_dict(torch.load(os.path.join(opt.save_dir,'last.pth')))
    if not opt.no_cuda:
        net.cuda()

    final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    plot_result={'train':[1.],'test':[1.]}
    confusion_mats = []

    for epoch in range(opt.epochs):
        t1 = time.time()
        confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
        # print('epoch:',epoch+1)
        np.random.shuffle(train_sequences[fold])
        net.train()
        for i, sequence in enumerate(train_sequences[fold], 1):

            signal=transformer.ToInputShape(signals[sequence],opt,test_flag =False)
            signal,label = transformer.ToTensor(signal,labels[sequence],no_cuda =opt.no_cuda)
                       
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
            if i%show_freq==0:       
                plot_result['train'].append(statistics.report(confusion_mat)[3])
                heatmap.draw(confusion_mat,opt,name = 'current_train')
                statistics.plotloss(plot_result,epoch+i/(train_sequences.shape[1]),opt)
                confusion_mat[:]=0

        plot_result,confusion_mat = evalnet(net,signals,labels,test_sequences[fold],epoch+1,plot_result)
        confusion_mats.append(confusion_mat)

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
    else:
        fold_final_confusion_mat += final_confusion_mat
        util.writelog('fold  -> macro-prec,reca,F1,err,kappa: '+str(statistics.report(final_confusion_mat)),opt,True)
        util.writelog('confusion_mat:\n'+str(final_confusion_mat)+'\n',opt,True)
        heatmap.draw(final_confusion_mat,opt,name = 'fold'+str(fold+1)+'_test')

if opt.k_fold != 1:
    statistics.statistics(fold_final_confusion_mat, opt, 'final', 'k-fold-final_test')

if opt.mergelabel:
    mat = statistics.mergemat(fold_final_confusion_mat, opt.mergelabel)
    statistics.statistics(mat, opt, 'merge', 'mergelabel_test')

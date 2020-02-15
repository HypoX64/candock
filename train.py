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

opt = Options().getparse()
localtime = time.asctime(time.localtime(time.time()))
util.writelog('\n\n'+str(localtime)+'\n'+str(opt))
t1 = time.time()

'''
change your own data to train
but the data needs meet the following conditions: 
1.type   numpydata  signals:np.float16  stages:np.int16
2.shape             signals:[?,3000]   stages:[?]
3.fs = 100Hz
4.input signal data should be normalized!!
  we recommend signal data normalized useing 5_95_th for each subject, 
  example: signals_normalized=transformer.Balance_individualized_differences(signals_origin, '5_95_th')
'''
signals_train,labels_train,signals_test,labels_test = dataloader.loaddataset(opt.dataset_dir,opt.dataset_name,opt.signal_name,opt.sample_num,opt.BID,opt.select_sleep_time)

util.writelog('train:',True)
stage_cnt,stage_cnt_per = statistics.stage(opt,labels_train)
util.writelog('test:',True)
_,_ = statistics.stage(opt,labels_test)
signals_train,labels_train = transformer.batch_generator(signals_train,labels_train,opt.batchsize)
signals_test,labels_test = transformer.batch_generator(signals_test,labels_test,opt.batchsize)


batch_length = len(signals_train)
print('length of batch:',batch_length)
show_freq = int(len(labels_train)/5)
t2 = time.time()
print('load data cost time: %.2f'% (t2-t1),'s')

net=CreatNet(opt)
util.show_paramsnumber(net)

weight = np.ones(opt.label)
if opt.weight_mod == 'avg_best':
    weight = np.log(1/stage_cnt_per)
    weight[2] = weight[2]+1
    weight = weight/np.median(weight)
    weight = np.clip(weight, 0.8, 2)
print('Loss_weight:',weight)
weight = torch.from_numpy(weight).float()
# print(net)
if not opt.no_cuda:
    net.cuda()
    weight = weight.cuda()
if opt.pretrained:
    net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth'))
if opt.continue_train:
    net.load_state_dict(torch.load('./checkpoints/last.pth'))
if not opt.no_cudnn:
    torch.backends.cudnn.benchmark = True

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.CrossEntropyLoss(weight)

def evalnet(net,signals,stages,epoch,plot_result={}):
    # net.eval()
    confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    for i, (signal,stage) in enumerate(zip(signals,stages), 1):

        signal=transformer.ToInputShape(signal,opt.model_name,test_flag =True)
        signal,stage = transformer.ToTensor(signal,stage,no_cuda =opt.no_cuda)
        with torch.no_grad():
            out = net(signal)
        pred = torch.max(out, 1)[1]

        pred=pred.data.cpu().numpy()
        stage=stage.data.cpu().numpy()
        for x in range(len(pred)):
            confusion_mat[stage[x]][pred[x]] += 1

    recall,acc,sp,err,k  = statistics.result(confusion_mat)
    plot_result['test'].append(err)   
    heatmap.draw(confusion_mat,opt.label_name,opt.label_name,name = 'test')
    print('recall,acc,sp,err,k: '+str(statistics.result(confusion_mat)))
    return plot_result,confusion_mat

print('begin to train ...')
final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)

plot_result={'train':[1.],'test':[1.]}
confusion_mats = []

for epoch in range(opt.epochs):
    t1 = time.time()
    confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
    print('epoch:',epoch+1)
    net.train()
    for i, (signal,stage) in enumerate(zip(signals_train,labels_train), 1):

        signal=transformer.ToInputShape(signal,opt.model_name,test_flag =False)
        signal,stage = transformer.ToTensor(signal,stage,no_cuda =opt.no_cuda)
                   
        out = net(signal)
        loss = criterion(out, stage)
        pred = torch.max(out, 1)[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred=pred.data.cpu().numpy()
        stage=stage.data.cpu().numpy()
        for x in range(len(pred)):
            confusion_mat[stage[x]][pred[x]] += 1
        if i%show_freq==0:       
            plot_result['train'].append(statistics.result(confusion_mat)[3])
            heatmap.draw(confusion_mat,opt.label_name,opt.label_name,name = 'train')
            statistics.show(plot_result,epoch+i/(batch_length*0.8))
            confusion_mat[:]=0

    plot_result,confusion_mat = evalnet(net,signals_test,labels_test,epoch+1,plot_result)
    confusion_mats.append(confusion_mat)
    # scheduler.step()

    torch.save(net.cpu().state_dict(),'./checkpoints/last.pth')
    if (epoch+1)%opt.network_save_freq == 0:
        torch.save(net.cpu().state_dict(),'./checkpoints/'+opt.model_name+'_epoch'+str(epoch+1)+'.pth')
        print('network saved.')
    if not opt.no_cuda:
        net.cuda()

    t2=time.time()
    if epoch+1==1:
        print('cost time: %.2f' % (t2-t1),'s')

pos = plot_result['test'].index(min(plot_result['test']))-1
final_confusion_mat = confusion_mats[pos]
util.writelog('final: '+'recall,acc,sp,err,k: '+str(statistics.result(final_confusion_mat)),True)
util.writelog('confusion_mat:\n'+str(final_confusion_mat),True)
statistics.stagefrommat(final_confusion_mat)
heatmap.draw(final_confusion_mat,opt.label_name,opt.label_name,name = 'final_test')
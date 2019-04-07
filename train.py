import numpy as np
import time
import util
import os
import time
import transformer
import dataloader
# import models
from creatnet import CreatNet
import torch
from torch import nn, optim
import statistics
import torch.backends.cudnn as cudnn
import heatmap
from options import Options
import warnings
warnings.filterwarnings("ignore")

opt = Options().getparse()
localtime = time.asctime(time.localtime(time.time()))
util.writelog('\n\n'+str(localtime)+'\n'+str(opt))

t1 = time.time()
signals,stages = dataloader.loaddataset(opt,opt.dataset_dir,opt.dataset_name,opt.signal_name,opt.sample_num,shuffle=True,BID=None)
stage_cnt,stage_cnt_per = statistics.stage(stages)
signals,stages = transformer.batch_generator(signals,stages,opt.batchsize,shuffle = True)
batch_length = len(stages)
print('length of batch:',batch_length)
train_sequences,test_sequences = transformer.k_fold_generator(batch_length,opt.fold_num)

show_freq = int(len(train_sequences[0])/5)
util.show_menory()
t2 = time.time()
print('load data cost time: %.2f'% (t2-t1),'s')

net=CreatNet(opt.model_name)
torch.save(net.cpu().state_dict(),'./checkpoints/'+opt.model_name+'.pth')

weight = np.array([1,1,1,1,1])
if opt.weight_mod == 'avg_best':
    weight = np.log(1/stage_cnt_per)
    weight[2] = weight[2]+1
    weight = np.clip(weight,1,5)
print('Loss_weight:',weight)
weight = torch.from_numpy(weight).float()
# print(net)
if not opt.no_cuda:
    net.cuda()
    weight = weight.cuda()
if not opt.no_cudnn:
    cudnn.benchmark = True

optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
criterion = nn.CrossEntropyLoss(weight)

def evalnet(net,signals,stages,sequences,epoch,plot_result={},mode = 'part'):
    # net.eval()
    if mode =='part':
        transformer.shuffledata(signals,stages)
        signals=signals[0:int(len(stages)/2)]
        stages=stages[0:int(len(stages)/2)]

    confusion_mat = np.zeros((5,5), dtype=int)
    for i, sequence in enumerate(sequences, 1):

        signal=transformer.ToInputShape(signals[sequence],opt.model_name,test_flag =True)
        signal,stage = transformer.ToTensor(signal,stages[sequence],no_cuda =opt.no_cuda)
        with torch.no_grad():
            out = net(signal)
        pred = torch.max(out, 1)[1]

        pred=pred.data.cpu().numpy()
        stage=stage.data.cpu().numpy()
        for x in range(len(pred)):
            confusion_mat[stage[x]][pred[x]] += 1
    if mode =='part':
        plot_result['test'].append(statistics.result(confusion_mat)[0])     
    else:
        recall,acc,error  = statistics.result(confusion_mat)
        plot_result['test'].append(recall)   
        heatmap.draw(confusion_mat,name = 'test')
        print('test avg_recall:','%.4f' % recall,'avg_acc:','%.4f' % acc,'error:','%.4f' % error)
        #util.writelog('epoch:'+str(epoch)+'  test avg_recall:'+str(round(recall,4))+'  avg_acc:'+str(round(acc,4))+'  error:'+str(round(error,4)))
    return plot_result,confusion_mat

print('begin to train ...')
final_confusion_mat = np.zeros((5,5), dtype=int)
for fold in range(opt.fold_num):
    net.load_state_dict(torch.load('./checkpoints/'+opt.model_name+'.pth'))
    if opt.pretrained:
        net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.model_name+'.pth'))
    if not opt.no_cuda:
        net.cuda()
    plot_result={'train':[0],'test':[0]}
    confusion_mats = []

    for epoch in range(opt.epochs):
        t1 = time.time()
        confusion_mat = np.zeros((5,5), dtype=int)
        print('fold:',fold+1,'epoch:',epoch+1)
        net.train()
        for i, sequence in enumerate(train_sequences[fold], 1):

            signal=transformer.ToInputShape(signals[sequence],opt.model_name,test_flag =False)
            signal,stage = transformer.ToTensor(signal,stages[sequence],no_cuda =opt.no_cuda)

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
                plot_result['train'].append(statistics.result(confusion_mat)[0])
                heatmap.draw(confusion_mat,name = 'train')
                # plot_result=evalnet(net,signals_eval,stages_eval,plot_result,show_freq,mode = 'part')
                statistics.show(plot_result,epoch+i/(batch_length*0.8))
                confusion_mat[:]=0

        plot_result,confusion_mat = evalnet(net,signals,stages,test_sequences[fold],epoch+1,plot_result,mode = 'all')
        confusion_mats.append(confusion_mat)
        # scheduler.step()

        if (epoch+1)%opt.network_save_freq == 0:
            torch.save(net.cpu().state_dict(),'./checkpoints/'+opt.model_name+'_epoch'+str(epoch+1)+'.pth')
            print('network saved.')
            if not opt.no_cuda:
                net.cuda()

        t2=time.time()
        print('cost time: %.2f' % (t2-t1),'s')
    pos = plot_result['test'].index(max(plot_result['test']))-1
    final_confusion_mat = final_confusion_mat+confusion_mats[pos]
    recall,acc,error  = statistics.result(confusion_mats[pos])
    print('\nfold:',fold+1,'finished',' avg_recall:','%.4f' % recall,'avg_acc:','%.4f' % acc,'error:','%.4f' % error,'\n')
    util.writelog('fold:'+str(fold+1)+'  test avg_recall:'+str(round(recall,4))+'  avg_acc:'+str(round(acc,4))+'  error:'+str(round(error,4)))
    util.writelog('confusion_mat:\n'+str(confusion_mat))

recall,acc,error  = statistics.result(final_confusion_mat)
#print('all finished!\n',final_confusion_mat)
#print('avg_recall:','%.4f' % recall,'avg_acc:','%.4f' % acc,'error:','%.4f' % error)
util.writelog('final:'+'  test avg_recall:'+str(round(recall,4))+'  avg_acc:'+str(round(acc,4))+'  error:'+str(round(error,4)),True)
util.writelog('confusion_mat:\n'+str(confusion_mat),True)
statistics.stagefrommat(confusion_mat)
heatmap.draw(final_confusion_mat,name = 'final_test')
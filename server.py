import os
import time
import shutil
import numpy as np
import torch
from torch import nn, optim
import warnings

from util import util,transformer,dataloader,statistics,plot,options
from util import array_operation as arr
from models import creatnet,io
from train import trainnet,evalnet

opt = options.Options()
opt.parser.add_argument('--ip',type=str,default='', help='')
opt = opt.getparse()
torch.cuda.set_device(opt.gpu_id)
opt.k_fold = 0
opt.save_dir = './datasets/server/tmp'

'''load ori data'''
signals,labels = dataloader.loaddataset(opt)
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
opt = options.get_auto_options(opt, label_cnt_per, label_num, signals.shape)

'''def network'''
net=creatnet.CreatNet(opt)
if opt.pretrained != '':
    net.load_state_dict(torch.load(opt.pretrained))
io.show_paramsnumber(net,opt)
if opt.gpu_id != -1:
    net.cuda()
    if not opt.no_cudnn:
        torch.backends.cudnn.benchmark = True
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
criterion_class = nn.CrossEntropyLoss(opt.weight)
criterion_auto = nn.MSELoss()


'''Receive data'''
if os.path.isdir('./datasets/server/data'):
    shutil.rmtree('./datasets/server/data')
os.system('unzip ./datasets/server/data.zip -d ./datasets/server/')
categorys = os.listdir('./datasets/server/data')
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
        signal = arr.normliaze(signal_ori,'5_95',truncated=4)
        for j in range(1,len(signal)//opt.loadsize-1):
            received_signals.append(signal[j*opt.loadsize:(j+1)*opt.loadsize])
            received_labels.append(i)

received_signals = np.array(received_signals).reshape(-1,opt.input_nc,opt.loadsize)
received_labels = np.array(received_labels).reshape(-1,1)

'''merge data'''
signals = signals[receive_category*500:]
labels = labels[receive_category*500:]
signals = np.concatenate((signals, received_signals))
labels = np.concatenate((labels, received_labels))
# print(received_signals.shape,received_labels.shape)
# print(signals.shape,labels.shape)

'''
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
train_sequences,test_sequences = transformer.k_fold_generator(len(labels),opt.k_fold)

final_confusion_mat = np.zeros((opt.label,opt.label), dtype=int)
confusion_mats = []
plot_result = {'train':[],'test':[]}
for epoch in range(opt.epochs):
    
    t1 = time.time()
    np.random.shuffle(train_sequences[fold])
    plot_result = trainnet(net,signals,labels,train_sequences[fold],epoch+1,plot_result)
    plot_result,confusion_mat_eval = evalnet(net,signals,labels,test_sequences[fold],epoch+1,plot_result)
    confusion_mats.append(confusion_mat_eval)

    torch.save(net.cpu().state_dict(),os.path.join(opt.save_dir,'last.pth'))
    if (epoch+1)%opt.network_save_freq == 0:
        torch.save(net.cpu().state_dict(),os.path.join(opt.save_dir,opt.model_name+'_epoch'+str(epoch+1)+'.pth'))
        print('network saved.')
    if opt.gpu_id != -1:
        net.cuda()

    t2=time.time()
    if epoch+1==1:
        util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)

# signals,labels = dataloader.loaddataset(opt)
# label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
'''
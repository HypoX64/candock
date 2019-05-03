import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import util
import transformer
import dataloader
from options import Options
from creatnet import CreatNet

'''
@hypox64
19/04/13
download pretrained model and test data here:
https://drive.google.com/open?id=1pup2_tZFGQQwB-hoXRjpMxiD4Vmpn0Lf
'''
opt = Options().getparse()
#choose and creat model
opt.model_name = 'micro_multi_scale_resnet_1d'
net=CreatNet(opt.model_name)

if not opt.no_cuda:
    net.cuda()
if not opt.no_cudnn:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

#load prtrained_model
net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth'))
net.eval()

def runmodel(eeg):
    eeg = eeg.reshape(1,-1)
    eeg = transformer.ToInputShape(eeg,opt.model_name,test_flag =True)
    eeg = transformer.ToTensor(eeg,no_cuda =opt.no_cuda)
    out = net(eeg)
    pred = torch.max(out, 1)[1]
    pred_stage=pred.data.cpu().numpy()
    return pred_stage[0]

'''
you can change your input data here.
but the data needs meet the following conditions: 
1.fs = 100Hz
2.collect by uv
3.type   numpydata  signals:np.float16  stages:np.int16
4.shape             signals:[?,3000]   stages:[?]
'''
eegdata = np.load('./datasets/simple_test/sleep_edfx_Fpz_Cz_test.npy')
true_stages = np.load('./datasets/simple_test/sleep_edfx_stages_test.npy')
print('shape of eegdata:',eegdata.shape)
print('shape of true_stage:',true_stages.shape)

#Normalize
eegdata = transformer.Balance_individualized_differences(eegdata, '5_95_th')

#run pretrained model
pred_stages=[]
for i in range(len(eegdata)):
    pred_stages.append(runmodel(eegdata[i]))
pred_stages = np.array(pred_stages)

print('err:',sum((true_stages[i]!=pred_stages[i])for i in range(len(pred_stages)))/len(true_stages)*100,'%')

#plot result
plt.subplot(211)
plt.plot(true_stages+1)
plt.xlim((0,len(true_stages)))
plt.ylim((0,6))
plt.yticks([1, 2, 3, 4, 5],['N3', 'N2', 'N1', 'REM', 'W'])
plt.xticks([],[])
plt.title('Manually scored hypnogram')

plt.subplot(212)
plt.plot(pred_stages+1)
plt.xlim((0,len(true_stages)))
plt.ylim((0,6))
plt.yticks([1, 2, 3, 4, 5],['N3', 'N2', 'N1', 'REM', 'W'])
plt.xlabel('Epoch number')
plt.title('Auto scored hypnogram')
plt.show()


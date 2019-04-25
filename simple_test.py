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
'''
opt = Options().getparse()
net=CreatNet(opt.model_name)

if not opt.no_cuda:
    net.cuda()
if not opt.no_cudnn:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
if opt.pretrained:
    net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.model_name+'.pth'))

# N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4
stage_map={0:'stage3',1:'stage2',2:'stage3',3:'REM',4:'Wake'}

def runmodel(eegdata):
    eegdata = eegdata.reshape(1,-1)
    eegdata = transformer.ToInputShape(eegdata,opt.model_name,test_flag =True)
    eegdata = transformer.ToTensor(eegdata,no_cuda =opt.no_cuda)
    with torch.no_grad():
        out = net(eegdata)
        pred = torch.max(out, 1)[1]
        pred_stage=pred.data.cpu().numpy()
    return pred_stage[0]


'''
you can change your input data here.
but the data needs meet the following conditions: 
1.record for 1 epoch(30s)
2.fs = 100Hz
3.uv
'''
eegdata = np.load('./datasets/simple_test_data.npy')
print('the shape of eegdata:',eegdata.shape)

stage = runmodel(eegdata)

print('the sleep stage of this signal is:',stage_map[stage])
plt.plot(eegdata)
plt.show()

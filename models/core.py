import os
import time

import numpy as np
import torch
from torch import nn, optim
from multiprocessing import Process, Queue
# import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")
from util import util,transformer,dataloader,statistics,plot
from . import creatnet

def show_paramsnumber(net,opt):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    util.writelog('net parameters: '+str(parameters)+'M',opt,True)

class Core(object):
    def __init__(self, opt):
        super(Core, self).__init__()
        self.opt = opt
        self.epoch = 1
        if self.opt.gpu_id != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)
            #torch.cuda.set_device(self.opt.gpu_id)
            if not self.opt.no_cudnn:
                torch.backends.cudnn.benchmark = True

    def network_init(self,printflag=False):

        self.net = creatnet.creatnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)
        self.criterion_class = nn.CrossEntropyLoss(self.opt.weight)
        self.criterion_auto = nn.MSELoss()
        self.epoch = 1
        self.plot_result = {'train':[],'eval':[],'F1':[]}
        self.confusion_mats = []
        self.test_flag = True

        if printflag:
            #util.writelog('network:\n'+str(self.net),self.opt,True)
            show_paramsnumber(self.net,self.opt)

        if self.opt.pretrained != '':
            self.net.load_state_dict(torch.load(self.opt.pretrained))
        if self.opt.continue_train:
            self.net.load_state_dict(torch.load(os.path.join(self.opt.save_dir,'last.pth')))
        if self.opt.gpu_id != -1:
            self.net.cuda()
    
    def save(self):
        torch.save(self.net.cpu().state_dict(),os.path.join(self.opt.save_dir,'last.pth'))
        if (self.epoch-1)%self.opt.network_save_freq == 0:
            torch.save(self.net.cpu().state_dict(),os.path.join(self.opt.save_dir,self.opt.model_name+'_epoch'+str(self.epoch-1)+'.pth'))
            print('network saved.')
        if self.opt.gpu_id != -1:
            self.net.cuda()

    def save_traced_net(self):
        self.net.cpu()
        self.net.eval()
        example = torch.rand(1,self.opt.input_nc, self.opt.finesize)
        traced_script_module = torch.jit.trace(self.net, example)
        traced_script_module.save(os.path.join(self.opt.save_dir,'model.pt'))
        print('Save traced network, example shape:',(1,self.opt.input_nc, self.opt.finesize))
        if self.opt.gpu_id != -1:
            self.net.cuda()

    def preprocessing(self,signals, labels, sequences):
        for i in range(len(sequences)//self.opt.batchsize):
            signal,label = transformer.batch_generator(signals, labels, sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize])
            signal = transformer.ToInputShape(signal,self.opt,test_flag =self.test_flag)
            self.queue.put([signal,label])

    def start_process(self,signals,labels,sequences):
        p = Process(target=self.preprocessing,args=(signals,labels,sequences))         
        p.daemon = True
        p.start()
    
    def process_pool_init(self,signals,labels,sequences):
        self.queue = Queue(self.opt.load_thread*2)
        process_batch_num = len(sequences)//self.opt.batchsize//self.opt.load_thread
        if process_batch_num == 0:
            print('\033[1;33m'+'Warning: too much load thread'+'\033[0m') 
            self.start_process(signals,labels,sequences)
        else:
            for i in range(self.opt.load_thread):
                if i != self.opt.load_thread-1:
                    self.start_process(signals,labels,sequences[i*self.opt.load_thread*self.opt.batchsize:(i+1)*self.opt.load_thread*self.opt.batchsize])
                else:
                    self.start_process(signals,labels,sequences[i*self.opt.load_thread*self.opt.batchsize:])

    def forward(self,signal,label,features,confusion_mat):
        if self.opt.model_name == 'autoencoder':
            out,feature = self.net(signal)
            loss = self.criterion_auto(out, signal)
            features[i*self.opt.batchsize:(i+1)*self.opt.batchsize,:self.opt.feature] = (feature.data.cpu().numpy()).reshape(self.opt.batchsize,-1)
            features[i*self.opt.batchsize:(i+1)*self.opt.batchsize,self.opt.feature] = label.data.cpu().numpy()
        else:
            out = self.net(signal)
            loss = self.criterion_class(out, label)
            pred = (torch.max(out, 1)[1]).data.cpu().numpy()
            label=label.data.cpu().numpy()
            for x in range(len(pred)):
                confusion_mat[label[x]][pred[x]] += 1
        return loss,features,confusion_mat

    def train(self,signals,labels,sequences):
        self.net.train()
        self.test_flag = False
        epoch_loss = 0
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        features = np.zeros((len(sequences)//self.opt.batchsize*self.opt.batchsize,self.opt.feature+1))
        
        np.random.shuffle(sequences)
        self.process_pool_init(signals, labels, sequences)
        for i in range(len(sequences)//self.opt.batchsize):
            signal,label = self.queue.get()
            signal,label = transformer.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            loss,features,confusion_mat=self.forward(signal, label, features, confusion_mat)

            epoch_loss += loss.item()     
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
        self.plot_result['train'].append(epoch_loss/i)
        plot.draw_loss(self.plot_result,self.epoch+i/(sequences.shape[0]/self.opt.batchsize),self.opt)
        # if self.opt.model_name != 'autoencoder':
        #     plot.draw_heatmap(confusion_mat,self.opt,name = 'current_train')


    def eval(self,signals,labels,sequences):
        self.test_flag = True
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        features = np.zeros((len(sequences)//self.opt.batchsize*self.opt.batchsize,self.opt.feature+1))
        epoch_loss = 0

        self.process_pool_init(signals, labels, sequences)
        for i in range(len(sequences)//self.opt.batchsize):
            signal,label = self.queue.get()
            signal,label = transformer.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            with torch.no_grad():
                loss,features,confusion_mat = self.forward(signal, label, features, confusion_mat)
                epoch_loss += loss.item()

        if self.opt.model_name != 'autoencoder':
            recall,acc,sp,err,k  = statistics.report(confusion_mat)         
            #plot.draw_heatmap(confusion_mat,self.opt,name = 'current_eval')
            print('epoch:'+str(self.epoch),' macro-prec,reca,F1,err,kappa: '+str(statistics.report(confusion_mat)))
            self.plot_result['F1'].append(statistics.report(confusion_mat)[2])
        else:
            plot.draw_autoencoder_result(signal.data.cpu().numpy(), out.data.cpu().numpy(),self.opt)
            print('epoch:'+str(self.epoch),' loss: '+str(round(epoch_loss/i,5)))
            plot.draw_scatter(features, self.opt)
        
        self.plot_result['eval'].append(epoch_loss/i) 

        self.epoch +=1
        self.confusion_mats.append(confusion_mat)
        # return confusion_mat


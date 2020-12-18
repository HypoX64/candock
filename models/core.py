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
from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
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
            # torch.cuda.set_device(self.opt.gpu_id)
            if not self.opt.no_cudnn:
                torch.backends.cudnn.benchmark = True

    def network_init(self,printflag=False):

        self.net = creatnet.creatnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)
        self.loss_classifier = nn.CrossEntropyLoss(self.opt.weight)
        self.loss_autoencoder = nn.MSELoss()
        self.loss_domain_c = torch.nn.NLLLoss(self.opt.weight)
        self.loss_domain_d = torch.nn.NLLLoss()
        self.epoch = 1
        self.plot_result = {'train':[],'eval':[],'F1':[],'err':[]}
        self.confusion_mats = []
        self.test_flag = True
        self.pre_labels = []
        self.save_sequences = []

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
        _times = np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)
        for i in range(_times):
            if i != _times-1:
                signal,label = transforms.batch_generator(signals, labels, sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize])
            else:
                signal,label = transforms.batch_generator(signals, labels, sequences[i*self.opt.batchsize:])
            signal = transforms.ToInputShape(self.opt,signal,test_flag =self.test_flag)
            self.queue.put([signal,label])

    def start_process(self,signals,labels,sequences):
        p = Process(target=self.preprocessing,args=(signals,labels,sequences))         
        p.daemon = True
        p.start()
    
    def process_pool_init(self,signals,labels,sequences):
        self.queue = Queue(self.opt.load_thread*2)
        process_batch_num = len(sequences)//self.opt.batchsize//self.opt.load_thread
        if process_batch_num == 0:
            if self.epoch == 1:
                print('\033[1;33m'+'Warning: too much load thread'+'\033[0m') 
            self.start_process(signals,labels,sequences)
        else:
            for i in range(self.opt.load_thread):
                if i != self.opt.load_thread-1:
                    self.start_process(signals,labels,sequences[i*self.opt.load_thread*self.opt.batchsize:(i+1)*self.opt.load_thread*self.opt.batchsize])
                else:
                    self.start_process(signals,labels,sequences[i*self.opt.load_thread*self.opt.batchsize:])

    def forward(self,signal,label,features,confusion_mat):
        if self.opt.mode == 'autoencoder':
            out,feature = self.net(signal)
            loss = self.loss_autoencoder(out, signal)
            label = label.data.cpu().numpy()
            feature = (feature.data.cpu().numpy()).reshape(self.opt.batchsize,-1)
            for i in range(self.opt.batchsize):
                features.append(np.concatenate((feature[i], [label[i]])))

        elif self.opt.mode in ['classify_1d','classify_2d']:
            out = self.net(signal)
            # print(out.size(), label.size())
            loss = self.loss_classifier(out, label)
            pred = (torch.max(out, 1)[1]).data.cpu().numpy()
            label = label.data.cpu().numpy()
            for x in range(len(pred)):
                confusion_mat[label[x]][pred[x]] += 1
                self.pre_labels.append(pred[x])

        elif self.opt.mode == 'domain':
            out,_ = self.net(signal,0)
            loss = self.loss_domain_c(out, label)
            pred = (torch.max(out, 1)[1]).data.cpu().numpy()
            label = label.data.cpu().numpy()
            for x in range(len(pred)):
                confusion_mat[label[x]][pred[x]] += 1
                self.pre_labels.append(pred[x])
        return out,loss,features,confusion_mat

    def train(self,signals,labels,sequences):
        self.net.train()
        self.test_flag = False
        features = []
        epoch_loss = 0
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        
        np.random.shuffle(sequences)
        self.process_pool_init(signals, labels, sequences)

        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            self.optimizer.zero_grad()

            signal,label = self.queue.get()   
            signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            output,loss,features,confusion_mat = self.forward(signal, label, features, confusion_mat)

            epoch_loss += loss.item()     
            loss.backward()
            self.optimizer.step()
       
        self.plot_result['train'].append(epoch_loss/(i+1))
        if self.epoch%10 == 0:
            plot.draw_loss(self.plot_result,self.epoch+(i+1)/(sequences.shape[0]/self.opt.batchsize),self.opt)
        # if self.opt.model_name != 'autoencoder':
        #     plot.draw_heatmap(confusion_mat,self.opt,name = 'current_train')


    def eval(self,signals,labels,sequences):
        self.test_flag = True
        self.pre_labels = []
        features = []
        epoch_loss = 0
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)

        np.random.shuffle(sequences)
        self.save_sequences = sequences
        self.process_pool_init(signals, labels, sequences)
        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            signal,label = self.queue.get()
            signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            with torch.no_grad():
                output,loss,features,confusion_mat = self.forward(signal, label, features, confusion_mat)
                epoch_loss += loss.item()

        if self.opt.mode == 'autoencoder':
            if self.epoch%10 == 0:
                plot.draw_autoencoder_result(signal.data.cpu().numpy(), output.data.cpu().numpy(),self.opt)
                print('epoch:'+str(self.epoch),' loss: '+str(round(epoch_loss/i,5)))
                plot.draw_scatter(features, self.opt)
        else:
            recall,acc,sp,err,k  = statistics.report(confusion_mat)         
            #plot.draw_heatmap(confusion_mat,self.opt,name = 'current_eval')
            print('epoch:'+str(self.epoch),' macro-prec,reca,F1,err,kappa: '+str(statistics.report(confusion_mat)))
            self.plot_result['F1'].append(statistics.report(confusion_mat)[2])
            self.plot_result['err'].append(statistics.report(confusion_mat)[3])
        
        self.plot_result['eval'].append(epoch_loss/(i+1)) 
        self.epoch +=1
        self.confusion_mats.append(confusion_mat)


    def dann_train(self,signals,labels,src_sequences,dst_sequences):
        self.net.train()
        self.test_flag = False
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        np.random.shuffle(src_sequences)
        self.process_pool_init(signals, labels, src_sequences)

        epoch_closs = 0;epoch_dloss = 0
        epoch_iter_length = np.ceil(len(src_sequences)/self.opt.batchsize).astype(np.int)
        dst_signals_len = len(dst_sequences)

        for i in range(epoch_iter_length):

            p = float(i + self.epoch * epoch_iter_length) / self.opt.epochs / epoch_iter_length
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # print(alpha)

            self.optimizer.zero_grad()

            s_signal,s_label = self.queue.get()
            this_batch_len = s_signal.shape[0]
            s_signal,s_label = transforms.ToTensor(s_signal,s_label,gpu_id =self.opt.gpu_id)
            s_domain = transforms.ToTensor(None,np.zeros(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)

            d_signal = signals[np.random.choice(np.arange(0, dst_signals_len),this_batch_len,replace=False)]
            d_signal = transforms.ToInputShape(self.opt,d_signal,test_flag =self.test_flag)
            d_signal = transforms.ToTensor(d_signal,None,gpu_id =self.opt.gpu_id)
            d_domain = transforms.ToTensor(None,np.ones(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
            
            class_output, domain_output = self.net(s_signal, alpha=alpha)
            # print(class_output.dtype, s_label.dtype,)
            loss_s_label = self.loss_domain_c(class_output, s_label)
            #print(domain_output, s_domain)
            loss_s_domain = self.loss_domain_d(domain_output, s_domain)
            _, domain_output = self.net(d_signal, alpha=alpha)
            #print(domain_output.size(), d_domain.size())
            loss_d_domain = self.loss_domain_d(domain_output, d_domain)
            loss = loss_s_label+loss_s_domain+loss_d_domain
            epoch_closs += loss_s_label.item()
            loss.backward()
            self.optimizer.step()
       
        self.plot_result['train'].append(epoch_closs/(i+1))
        if self.epoch%10 == 0:
            plot.draw_loss(self.plot_result,self.epoch+(i+1)/(src_sequences.shape[0]/self.opt.batchsize),self.opt)


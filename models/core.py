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
        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_id
            if not self.opt.no_cudnn:
                torch.backends.cudnn.benchmark = True

    def network_init(self,printflag=False):
        # Network & Optimizer & loss
        self.net = creatnet.creatnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)
        self.loss_classifier = nn.CrossEntropyLoss(self.opt.weight)
        if self.opt.mode == 'autoencoder':
            self.loss_autoencoder = nn.MSELoss()
        elif self.opt.mode == 'domain':
            self.loss_dann_c = torch.nn.NLLLoss(self.opt.weight)
            self.loss_dann_d = torch.nn.NLLLoss()
            self.loss_rd_true_domain = nn.CrossEntropyLoss()
            self.loss_rd_conf_domain = nn.CrossEntropyLoss()

        # save stack
        self.epoch = 1
        self.plot_result = {'train':[],'eval':[],'F1':[],'err':[]}
        self.confusion_mats = []
        self.test_flag = True
        self.eval_detail = [[],[],[]] # sequences, ture_labels, pre_labels

        if printflag:
            #util.writelog('network:\n'+str(self.net),self.opt,True)
            show_paramsnumber(self.net,self.opt)

        if self.opt.pretrained != '':
            self.net.load_state_dict(torch.load(self.opt.pretrained))
        if self.opt.continue_train:
            self.net.load_state_dict(torch.load(os.path.join(self.opt.save_dir,'last.pth')))
        if self.opt.gpu_id != '-1' and len(self.opt.gpu_id) == 1:
            self.net.cuda()
        elif self.opt.gpu_id != '-1' and len(self.opt.gpu_id) > 1:
            self.net = nn.DataParallel(self.net)
            self.net.cuda()
    
    def save(self):
        if self.opt.gpu_id == '-1' or len(self.opt.gpu_id) == 1:
            torch.save(self.net.cpu().state_dict(),os.path.join(self.opt.save_dir,'last.pth'))
        else:
            torch.save(self.net.module.cpu().state_dict(),os.path.join(self.opt.save_dir,'last.pth'))

        if (self.epoch-1)%self.opt.network_save_freq == 0:
            if self.opt.gpu_id == '-1' or len(self.opt.gpu_id) == 1:
                torch.save(self.net.cpu().state_dict(),os.path.join(self.opt.save_dir,self.opt.model_name+'_epoch'+str(self.epoch-1)+'.pth'))
            else:
                torch.save(self.net.module.cpu().state_dict(),os.path.join(self.opt.save_dir,self.opt.model_name+'_epoch'+str(self.epoch-1)+'.pth'))
            print('network saved.')

        if self.opt.gpu_id != '-1':
            self.net.cuda()

    def save_traced_net(self):
        self.net.cpu()
        self.net.eval()
        example = torch.rand(1,self.opt.input_nc, self.opt.finesize)
        if self.opt.gpu_id == '-1' or len(self.opt.gpu_id) == 1:
            traced_script_module = torch.jit.trace(self.net, example)
        else:
            traced_script_module = torch.jit.trace(self.net.module, example)
        traced_script_module.save(os.path.join(self.opt.save_dir,'model.pt'))
        print('Save traced network, example shape:',(1,self.opt.input_nc, self.opt.finesize))
        if self.opt.gpu_id != '-1':
            self.net.cuda()

    def preprocessing(self,signals, labels, sequences):
        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            signal,label = transforms.batch_generator(signals, labels, sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize])
            signal = transforms.ToInputShape(self.opt,signal,test_flag =self.test_flag)
            self.queue.put([signal,label,sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize]])

    def start_process(self,signals,labels,sequences):
        p = Process(target=self.preprocessing,args=(signals,labels,sequences))         
        p.daemon = True
        p.start()
    
    def process_pool_init(self,signals,labels,sequences):
        self.queue = Queue(self.opt.load_thread*2)
        load_thread = self.opt.load_thread
        process_batch_num = len(sequences)/self.opt.batchsize/load_thread
        if process_batch_num < 1:
            if self.epoch == 1:
                load_thread = len(sequences)//self.opt.batchsize
                process_batch_num = len(sequences)/self.opt.batchsize/load_thread
                print('\033[1;33m'+'Warning: too much load thread, try : '+str(load_thread)+'\033[0m') 

        for i in range(load_thread):
            if i != load_thread-1:
                self.start_process(signals,labels,sequences[int(i*process_batch_num)*self.opt.batchsize:int((i+1)*process_batch_num)*self.opt.batchsize])
            else:
                self.start_process(signals,labels,sequences[int(i*process_batch_num)*self.opt.batchsize:])

    def forward(self,signal,label,features,confusion_mat):
        if self.opt.mode == 'autoencoder':
            out,feature = self.net(signal)
            loss = self.loss_autoencoder(out, signal)
            label = label.data.cpu().numpy()
            feature = (feature.data.cpu().numpy()).reshape(self.opt.batchsize,-1)
            for i in range(self.opt.batchsize):
                features.append(np.concatenate((feature[i], [label[i]])))

        elif self.opt.mode in ['classify_1d','classify_2d','domain']:
            if self.opt.model_name in ['dann','dann_mobilenet']:
                out, _ = self.net(signal,0)
                loss = self.loss_dann_c(out, label)
            elif self.opt.model_name in ['rd_mobilenet']:
                out, _ = self.net(signal)
                loss = self.loss_classifier(out, label)
            else:
                out = self.net(signal)
                loss = self.loss_classifier(out, label)
            self.add_to_confusion_mat(label,out,confusion_mat,True)
        return out,loss,features,confusion_mat

    def add_to_confusion_mat(self,true_labels, pre_labels, confusion_mat, save_to_detail=False):
        pre_labels = (torch.max(pre_labels, 1)[1]).data.cpu().numpy()
        true_labels = true_labels.data.cpu().numpy()
        for x in range(len(pre_labels)):
            confusion_mat[true_labels[x]][pre_labels[x]] += 1
            if save_to_detail and self.test_flag:
                self.eval_detail[2].append(pre_labels[x])

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

            signal,label,_ = self.queue.get()  
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
        self.net.eval()
        self.test_flag = True
        self.eval_detail = [[],[],[]]
        features = []
        epoch_loss = 0
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)

        np.random.shuffle(sequences)
        self.save_sequences = sequences
        self.process_pool_init(signals, labels, sequences)
        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            signal,label,sequence = self.queue.get()
            self.eval_detail[0].append(list(sequence))
            self.eval_detail[1].append(list(label))
            signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            # with torch.no_grad():
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

    def dann_train(self,signals,labels,src_sequences,dst_sequences,beta=1.0):
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
            alpha = beta*(2. / (1. + np.exp(-10 * p)) - 1)
            self.optimizer.zero_grad()
            s_signal,s_label,sequence = self.queue.get()
            this_batch_len = s_signal.shape[0]
            s_signal,s_label = transforms.ToTensor(s_signal,s_label,gpu_id =self.opt.gpu_id)
            s_domain = transforms.ToTensor(None,np.zeros(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)

            d_signal = signals[np.random.choice(np.arange(0, dst_signals_len),this_batch_len,replace=False)]
            d_signal = transforms.ToInputShape(self.opt,d_signal,test_flag =self.test_flag)
            d_signal = transforms.ToTensor(d_signal,None,gpu_id =self.opt.gpu_id)
            d_domain = transforms.ToTensor(None,np.ones(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
            
            class_output, domain_output = self.net(s_signal, alpha=alpha)
            loss_s_label = self.loss_dann_c(class_output, s_label)
            loss_s_domain = self.loss_dann_d(domain_output, s_domain)
            _, domain_output = self.net(d_signal, alpha=alpha)
            loss_d_domain = self.loss_dann_d(domain_output, d_domain)
            loss = loss_s_label+loss_s_domain+loss_d_domain
            epoch_closs += loss_s_label.item()
            loss.backward()
            self.optimizer.step()
       
        self.plot_result['train'].append(epoch_closs/(i+1))
        if self.epoch%10 == 0:
            plot.draw_loss(self.plot_result,self.epoch+(i+1)/(src_sequences.shape[0]/self.opt.batchsize),self.opt)

    def rd_train(self,signals,labels,sequences,alpha=1.0,beta=1.0):
        self.net.train()
        self.test_flag = False
        features = []
        epoch_loss = 0
        confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        
        np.random.shuffle(sequences)
        self.process_pool_init(signals, labels, sequences)

        # load domain
        domains = np.load(os.path.join(self.opt.dataset_dir,'domains.npy'))
        domains = dataloader.rebuild_domain(domains)

        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            self.optimizer.zero_grad()

            signal,label,sequence = self.queue.get()
            domain = transforms.batch_generator(None, domains, sequence)
            np.random.shuffle(sequence)
            conf_domain = transforms.batch_generator(None, domains, sequence)

            signal,label = transforms.ToTensor(signal,label,gpu_id=self.opt.gpu_id)
            domain = transforms.ToTensor(None,domain,gpu_id=self.opt.gpu_id)
            conf_domain = transforms.ToTensor(None,conf_domain,gpu_id=self.opt.gpu_id)

            class_output, domain_output = self.net(signal)
            loss_c = self.loss_classifier(class_output,label)
            loss_td = self.loss_rd_true_domain(domain_output,domain)
            loss_cd = self.loss_rd_conf_domain(domain_output,conf_domain)
            loss = alpha*loss_c + beta*(loss_cd+loss_td)

            epoch_loss += loss.item()     
            loss.backward()
            self.optimizer.step()
       
        self.plot_result['train'].append(epoch_loss/(i+1))
        if self.epoch%10 == 0:
            plot.draw_loss(self.plot_result,self.epoch+(i+1)/(sequences.shape[0]/self.opt.batchsize),self.opt)
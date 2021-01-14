import os
import time

import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue
# import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import sys
from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
from models import creatnet

def show_paramsnumber(net,opt):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    util.writelog('net parameters: '+str(parameters)+'M',opt,True,True)

class Core(object):
    def __init__(self, opt):
        super(Core, self).__init__()
        self.opt = opt
        self.fold = 0
        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_id
            if not self.opt.no_cudnn:
                torch.backends.cudnn.benchmark = True

    def network_init(self,printflag=False):
        # Network & Optimizer & loss
        self.net,self.exp = creatnet.creatnet(self.opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.opt.lr)
        self.loss_classifier = nn.CrossEntropyLoss(self.opt.weight)
        if self.opt.mode == 'autoencoder':
            self.loss_autoencoder = nn.MSELoss()
        elif self.opt.mode == 'domain':
            self.loss_dann_c = torch.nn.NLLLoss(self.opt.weight)
            self.loss_dann_d = torch.nn.NLLLoss()

        # save stack init
        self.step = 0
        self.epoch = 0
        self.procs = []
        self.features = []
        self.results = {'F1':[],'err':[],'loss':[],'confusion_mat':[],'eval_detail':[],'best_epoch':0}

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
    
    def save(self,net,path):
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.cpu().state_dict(),path)
        else:
            torch.save(net.cpu().state_dict(),path) 
        if self.opt.gpu_id != '-1':
            net.cuda()

    def epoch_save(self):
        self.save(self.net, os.path.join(self.opt.save_dir,'last.pth'))
        if (self.epoch)%self.opt.network_save_freq == 0:
            os.rename(os.path.join(self.opt.save_dir,'last.pth'), os.path.join(self.opt.save_dir,self.opt.model_name+'_epoch'+str(self.epoch)+'.pth'))
            print('network saved.')
        if self.opt.gpu_id != '-1':
            self.net.cuda()

    def save_traced_net(self):
        self.net.cpu()
        self.net.eval()
        if self.opt.gpu_id == '-1' or len(self.opt.gpu_id) == 1:
            traced_script_module = torch.jit.trace(self.net, self.exp)
        else:
            traced_script_module = torch.jit.trace(self.net.module, self.exp)
        traced_script_module.save(os.path.join(self.opt.save_dir,'model.pt'))
        print('Save traced network, example shape:',self.exp.size())
        if self.opt.gpu_id != '-1':
            self.net.cuda()

    def check_remain_time(self):
        if self.fold == 0 and self.epoch == 1:
            self.start_time = time.time()
        if self.fold == 0 and self.epoch == 2:
            util.writelog('-> pre epoch cost time : '+str(round(time.time()-self.start_time,2))+'s',self.opt,True,True)
        if (self.fold == 0 and self.epoch > 1) or self.fold != 0:
            v = (self.fold*self.opt.epochs+self.epoch-1)/(time.time()-self.start_time)
            remain = (self.opt.k_fold*self.opt.epochs-(self.fold*self.opt.epochs+self.epoch))/v
            self.opt.TBGlobalWriter.add_scalar('RemainTime',remain/3600,self.fold*self.opt.epochs+self.epoch)

    def preprocess(self,signals, labels, sequences, queue):
        for i in range(np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)):
            signal,label = transforms.batch_generator(signals, labels, sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize])
            signal = transforms.ToInputShape(self.opt,signal,test_flag =self.test_flag)
            queue.put([signal,label,sequences[i*self.opt.batchsize:(i+1)*self.opt.batchsize]])

    def start_process(self,signals,labels,sequences,queue):
        p = Process(target=self.preprocess,args=(signals,labels,sequences,queue))
        self.procs.append(p)
        p.daemon = True
        p.start()
    
    def load_pool_init(self,signals,labels,sequences):
        self.queue = Queue(self.opt.load_thread)
        self.load_thread = self.opt.load_thread
        process_batch_num = len(sequences)/self.opt.batchsize/self.load_thread
        if process_batch_num < 1:
            if self.epoch == 0:
                self.load_thread = np.clip(len(sequences)//self.opt.batchsize,1,256)
                process_batch_num = len(sequences)/self.opt.batchsize/self.load_thread
                print('\033[1;33m'+'Warning: too much load thread, try : '+str(self.load_thread)+'\033[0m') 
        for i in range(self.load_thread):
            if i != self.load_thread-1:
                self.start_process(signals,labels,sequences[int(i*process_batch_num)*self.opt.batchsize:int((i+1)*process_batch_num)*self.opt.batchsize],self.queue)
            else:
                self.start_process(signals,labels,sequences[int(i*process_batch_num)*self.opt.batchsize:],self.queue)
    
    def load_poll_terminate(self):
        for p in self.procs:
            if p.is_alive():
                p.terminate()
            p.join()
        self.procs = []

    def add_label_to_confusion_mat(self,true_labels, pre_labels, save_to_detail=False):
        pre_labels = (torch.max(pre_labels, 1)[1]).data.cpu().numpy()
        true_labels = true_labels.data.cpu().numpy()
        for x in range(len(pre_labels)):
            self.confusion_mat[true_labels[x]][pre_labels[x]] += 1
            if save_to_detail and self.test_flag:
                self.eval_detail['pre_labels'].append(pre_labels[x])
    
    def add_class_acc_to_tensorboard(self,tag):
        self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/F1', {tag:statistics.report(self.confusion_mat)[2]}, self.step)
        self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/Top1.err', {tag:statistics.report(self.confusion_mat)[3]}, self.step)
        
    def epoch_forward_init(self,signals,labels,sequences,istrain=True):
        if istrain:
            self.net.train()
            self.test_flag = False
        else:
            # self.net.eval()
            self.test_flag = True
        self.eval_detail = {'sequences':[],'ture_labels':[],'pre_labels':[]} # sequences, ture_labels, pre_labels
        self.features = []
        self.epoch_loss = 0
        self.confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
        np.random.shuffle(sequences)
        if len(sequences)%self.opt.batchsize==1:# drop_last=True when last batchsize =1
            sequences = sequences[:self.opt.batchsize*(len(sequences)%self.opt.batchsize)]
        self.load_pool_init(signals, labels, sequences)
        self.epoch_iter_length = np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)
    
    def forward(self,signal,label):
        if self.opt.mode == 'autoencoder':
            out,feature = self.net(signal)
            loss = self.loss_autoencoder(out, signal)
            label = label.data.cpu().numpy()
            feature = (feature.data.cpu().numpy()).reshape(self.opt.batchsize,-1)
            for i in range(self.opt.batchsize):
                self.features.append(np.concatenate((feature[i], [label[i]])))

        elif self.opt.mode in ['classify_1d','classify_2d','domain']:
            if self.opt.model_name in ['dann','dann_base']:
                out, _ = self.net(signal,0)
                loss = self.loss_dann_c(out, label)
            else:
                out = self.net(signal)
                loss = self.loss_classifier(out, label)
            self.add_label_to_confusion_mat(label,out,True)
        return out,loss

    def eval(self,signals,labels,sequences):
        self.epoch_forward_init(signals,labels,sequences,False)
        for i in range(self.epoch_iter_length):
            signal,label,sequence = self.queue.get()
            self.eval_detail['sequences'].append(list(sequence))
            self.eval_detail['ture_labels'].append(list(label))
            signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            with torch.no_grad():
                output,loss = self.forward(signal, label)
                self.epoch_loss += loss.item()

        if self.opt.mode == 'autoencoder':
            if (self.epoch+1)%10 == 0:
                plot.draw_autoencoder_result(signal.data.cpu().numpy(), output.data.cpu().numpy(),self.opt)
                plot.draw_scatter(self.features, self.opt)
        else:
            prec,reca,f1,err,kappa = statistics.report(self.confusion_mat)
            print('epoch:'+str(self.epoch+1),' macro-prec,reca,F1,err,kappa: '+str(statistics.report(self.confusion_mat)))
            self.add_class_acc_to_tensorboard('eval')
            self.results['F1'].append(f1)
            self.results['err'].append(err)
            self.results['confusion_mat'].append(self.confusion_mat)
            self.results['loss'].append(self.epoch_loss/(i+1))
            self.results['eval_detail'].append(self.eval_detail)
            if self.opt.best_index == 'f1':
                if f1 >= max(self.results['F1']):self.results['best_epoch'] = self.epoch
            elif self.opt.best_index == 'err':
                if err <= max(self.results['err']):self.results['best_epoch'] = self.epoch
        
        self.load_poll_terminate()  
        self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'eval_loss':self.epoch_loss/(i+1)}, self.step)
        self.epoch +=1
        
    def train(self,signals,labels,sequences):
        self.epoch_forward_init(signals,labels,sequences,True)
        for i in range(self.epoch_iter_length):
            signal,label,_ = self.queue.get()
            self.step = float(i/self.epoch_iter_length + self.epoch)
            self.optimizer.zero_grad()
            signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
            output,loss = self.forward(signal, label)
            self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'train_loss':loss.item()}, self.step)
            loss.backward()
            self.optimizer.step()
        self.load_poll_terminate()
        self.add_class_acc_to_tensorboard('train')

    def domain_random_loader_init(self,signals,labels,sequences,queue):
        pre_thread_iter = self.epoch_iter_length//self.load_thread
        last_thread_iter = self.epoch_iter_length - pre_thread_iter*(self.load_thread-1)
        for i in range(self.load_thread):
            if i != self.load_thread-1:
                self.start_process(signals,labels,np.random.choice(sequences,pre_thread_iter*self.opt.batchsize,replace=True),queue)
            else:
                self.start_process(signals,labels,np.random.choice(sequences,last_thread_iter*self.opt.batchsize,replace=True),queue)

    def dann_train(self,signals,labels,src_sequences,dst_sequences,beta=1.0,stable=False):
        self.epoch_forward_init(signals,labels,src_sequences,True)
        dst_signals_len = len(dst_sequences)
        domain_queue = Queue(self.opt.load_thread)
        self.domain_random_loader_init(signals,labels,dst_sequences,domain_queue)
        domains = np.load(os.path.join(self.opt.dataset_dir,'domains.npy'))
        domains = dataloader.rebuild_domain(domains)

        for i in range(self.epoch_iter_length):
            self.step = float(i/self.epoch_iter_length + self.epoch)
            p = self.step / self.opt.epochs
            if stable:alpha = beta
            else:alpha = beta * (2. / (1. + np.exp(-10 * p)) - 1)
            self.optimizer.zero_grad()
            # src
            s_signal,s_label,sequence = self.queue.get()
            this_batch_len = s_signal.shape[0]
            s_signal,s_label = transforms.ToTensor(s_signal,s_label,gpu_id =self.opt.gpu_id)
            if self.opt.domain_num == 2:
                s_domain = transforms.ToTensor(None,np.zeros(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
            else:
                s_domain = transforms.batch_generator(None, domains, sequence)
                s_domain = transforms.ToTensor(None,s_domain,gpu_id =self.opt.gpu_id)
            # dst
            d_signal,_,sequence = domain_queue.get()
            d_signal = transforms.ToTensor(d_signal[:this_batch_len],None,gpu_id =self.opt.gpu_id)
            if self.opt.domain_num == 2:
                d_domain = transforms.ToTensor(None,np.ones(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
            else:          
                d_domain = transforms.batch_generator(None, domains, sequence[:this_batch_len])
                d_domain = transforms.ToTensor(None,d_domain,gpu_id=self.opt.gpu_id)
           
            class_output, domain_output = self.net(s_signal, alpha=alpha)
            self.add_label_to_confusion_mat(s_label,class_output,False)
            loss_s_label = self.loss_dann_c(class_output, s_label)
            loss_s_domain = self.loss_dann_d(domain_output, s_domain)
            _, domain_output = self.net(d_signal, alpha=alpha)
            loss_d_domain = self.loss_dann_d(domain_output, d_domain)
            loss = loss_s_label+loss_s_domain+loss_d_domain
            self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'src_label':loss_s_label.item(),
                                                                'src_domain':loss_s_domain.item(),
                                                                'dst_domain':loss_d_domain.item()}, self.step)
            loss.backward()
            self.optimizer.step()

        self.add_class_acc_to_tensorboard('train')
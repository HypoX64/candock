import os
import time

import numpy as np
import torch
from torch import nn, optim
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue
# import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

import sys
from util import util,plot,options
from data import augmenter,transforms,dataloader,statistics
from models import creatnet,model_util

class Core(object):
    def __init__(self, opt):
        super(Core, self).__init__()
        self.opt = opt
        self.fold = 0
        self.n_epochs = self.opt.n_epochs
        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_id
            if not self.opt.no_cudnn:
                torch.backends.cudnn.benchmark = True

    def network_init(self,printflag=False):
        # Network & Optimizer & loss
        self.net,self.exp = creatnet.creatnet(self.opt)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.opt.lr)
        if self.opt.mode in ['domain','domain_1d','classify_1d','classify_2d']:
            self.loss_classifier = nn.CrossEntropyLoss(self.opt.weight)
        if self.opt.mode in ['domain','domain_1d']:
            self.loss_dann_c = torch.nn.NLLLoss(self.opt.weight)
            self.loss_dann_d = torch.nn.NLLLoss()
        if self.opt.mode =='dml':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low = 0)
            self.loss_func_dml = losses.TripletMarginLoss(margin=self.opt.margin, distance = distance, reducer = reducer)
            self.mining_func = miners.TripletMarginMiner(margin=self.opt.margin, distance = distance, type_of_triplets = "semihard")
            self.accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
            self.dml_tester = testers.BaseTester(batch_size=np.ceil(self.opt.batchsize/2))
            self.best_precision_at_1 = 0.0
            
        # save stack init
        self.step = 0
        self.epoch = 0
        self.procs = []
        self.features = []
        self.results = {'F1':[],'err':[],'loss':[],'confusion_mat':[],'eval_detail':[],'best_epoch':0}

        if printflag:
            #util.writelog('network:\n'+str(self.net),self.opt,True)
            model_util.show_paramsnumber(self.net,self.opt)

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
            self.net.cuda()

    def epoch_save(self,name='last.pth'):
        self.save(self.net, os.path.join(self.opt.save_dir,name))
        if (self.epoch)%self.opt.network_save_freq == 0:
            os.rename(os.path.join(self.opt.save_dir,name), os.path.join(self.opt.save_dir,self.opt.model_name+'_epoch'+str(self.epoch)+'.pth'))
            print('network saved.')

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
            v = (self.fold*self.n_epochs+self.epoch-1)/(time.time()-self.start_time)
            remain = (self.opt.k_fold*self.n_epochs-(self.fold*self.n_epochs+self.epoch))/v
            self.opt.TBGlobalWriter.add_scalar('RemainTime',remain/3600,self.fold*self.n_epochs+self.epoch)


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
    
    def epoch_init(self,istrain=True):
        if istrain:
            self.net.train()
            self.test_flag = False
        else:
            self.test_flag = True
        self.eval_detail = {'sequences':[],'ture_labels':[],'pre_labels':[]} # sequences, ture_labels, pre_labels
        self.features = []
        self.epoch_loss = 0
        self.confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)

    # def epoch_forward_init(self,signals,labels,sequences,istrain=True):
    #     if istrain:
    #         self.net.train()
    #         self.test_flag = False
    #     else:
    #         # self.net.eval()
    #         self.test_flag = True
    #     self.eval_detail = {'sequences':[],'ture_labels':[],'pre_labels':[]} # sequences, ture_labels, pre_labels
    #     self.features = []
    #     self.epoch_loss = 0
    #     self.confusion_mat = np.zeros((self.opt.label,self.opt.label), dtype=int)
    #     np.random.shuffle(sequences)
    #     if len(sequences)%self.opt.batchsize==1:# drop_last=True when last batchsize =1
    #         sequences = sequences[:self.opt.batchsize*(len(sequences)//self.opt.batchsize)]
    #     self.load_pool_init(signals, labels, sequences)
    #     self.epoch_iter_length = np.ceil(len(sequences)/self.opt.batchsize).astype(np.int)
    
    # def forward(self,signal,label):
    #     if self.opt.mode in ['classify_1d','classify_2d','domain','domain_1d']:
    #         if self.opt.mode in ['domain','domain_1d']:
    #             out, _ = self.net(signal,0)
    #             loss = self.loss_dann_c(out, label)
    #         else:
    #             out = self.net(signal)
    #             loss = self.loss_classifier(out, label)
    #         self.add_label_to_confusion_mat(label,out,True)
    #     return out,loss

    # def eval(self,signals,labels,sequences):
    #     # self.epoch_forward_init(signals,labels,sequences,False)
    #     if self.opt.debug:
    #         domains = np.load(os.path.join(self.opt.dataset_dir,'domains.npy'))
    #         print(domains[sequences])
    #     for i in range(self.epoch_iter_length):
    #         signal,label,sequence = self.queue.get()
    #         self.eval_detail['sequences'].append(list(sequence))
    #         self.eval_detail['ture_labels'].append(list(label))
    #         signal,label = transforms.ToTensor(signal,label,gpu_id =self.opt.gpu_id)
    #         with torch.no_grad():
    #             output,loss = self.forward(signal, label)
    #             self.epoch_loss += loss.item()

    #     prec,reca,f1,err,kappa = statistics.report(self.confusion_mat)
    #     util.writelog('epoch:'+str(self.epoch+1)+' macro-prec,reca,F1,err,kappa: '+str(statistics.report(self.confusion_mat)),self.opt,True)
    #     self.add_class_acc_to_tensorboard('eval')
    #     self.results['F1'].append(f1)
    #     self.results['err'].append(err)
    #     self.results['confusion_mat'].append(self.confusion_mat)
    #     self.results['loss'].append(self.epoch_loss/(i+1))
    #     self.results['eval_detail'].append(self.eval_detail)
    #     if self.opt.best_index == 'f1':
    #         if f1 >= max(self.results['F1']):self.results['best_epoch'] = self.epoch
    #     elif self.opt.best_index == 'err':
    #         if err <= min(self.results['err']):self.results['best_epoch'] = self.epoch
        
    #     # self.load_poll_terminate()  
    #     self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'eval_loss':self.epoch_loss/(i+1)}, self.step)
    #     self.epoch +=1

    def train(self,loader):
        self.epoch_init(istrain=True)
        for i, data in enumerate(loader):
            self.step = float(i/len(loader) + self.epoch)
            signal,label = data
            signal,label = transforms.ToDevice(signal,self.opt.gpu_id),transforms.ToDevice(label,self.opt.gpu_id)
            out = self.net(signal)
            loss = self.loss_classifier(out, label)
            self.add_label_to_confusion_mat(label,out,False)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'train_loss':self.epoch_loss/(i+1)}, self.step)
            self.add_class_acc_to_tensorboard('train')
    
    def eval(self,loader):
        self.epoch_init(istrain=False)
        for i, data in enumerate(loader):
            signal,label = data
            signal,label = transforms.ToDevice(signal,self.opt.gpu_id),transforms.ToDevice(label,self.opt.gpu_id)
            with torch.no_grad():
                out = self.net(signal)
                loss = self.loss_classifier(out, label)
            self.epoch_loss += loss.item()
            self.add_label_to_confusion_mat(label,out,False)

        prec,reca,f1,err,kappa = statistics.report(self.confusion_mat)
        util.writelog('epoch:'+str(self.epoch+1)+' macro-prec,reca,F1,err,kappa: '+str(statistics.report(self.confusion_mat)),self.opt,True)
        self.add_class_acc_to_tensorboard('eval')
        self.results['F1'].append(f1)
        self.results['err'].append(err)
        self.results['confusion_mat'].append(self.confusion_mat)
        self.results['loss'].append(self.epoch_loss/(i+1))
        self.results['eval_detail'].append(self.eval_detail)
        if self.opt.best_index == 'f1':
            if f1 >= max(self.results['F1']):self.results['best_epoch'] = self.epoch
        elif self.opt.best_index == 'err':
            if err <= min(self.results['err']):self.results['best_epoch'] = self.epoch
 
        self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'eval_loss':self.epoch_loss/(i+1)}, self.step)
        self.epoch +=1

    ####################################### Deep Metric Learning #######################################
    def dml_train(self,loader):
        self.epoch_init(istrain=True)
        # self.train_embeddings = None
        # self.train_labels = None
        for i, data in enumerate(loader):
            self.step = float(i/len(loader) + self.epoch)
            signal,label = data
            # _batchsize = len(signal)
            signal,label = transforms.ToDevice(signal,self.opt.gpu_id),transforms.ToDevice(label,self.opt.gpu_id)
            embedding = self.net(signal)
            indices_tuple = self.mining_func(embedding, label)
            loss = self.loss_func_dml(embedding, label, indices_tuple)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def dml_eval(self,train_dataset,eval_dataset):
        self.train_embeddings, self.train_labels = self.dml_tester.get_all_embeddings(train_dataset, self.net)
        self.test_embeddings, self.test_labels = self.dml_tester.get_all_embeddings(eval_dataset, self.net)
        accuracies = self.accuracy_calculator.get_accuracy( self.test_embeddings, 
                                                            self.train_embeddings,
                                                            self.test_labels,
                                                            self.train_labels,
                                                            False)
        accuracy = accuracies["precision_at_1"]
        if accuracy>self.best_precision_at_1:
            self.save(self.net, os.path.join(self.opt.save_dir,'best.pth'))
            self.best_precision_at_1 = accuracy
        print('epoch:'+str(self.epoch+1)+" Eval set accuracy (Precision@1) = {}".format(accuracy))
        self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/Recall@1', {'eval':accuracy}, self.step)
        plot.draw_dml(self.opt,self.test_embeddings,self.test_labels,self.epoch,10)
        np.save(os.path.join(self.opt.save_dir,'test_embeddings'),self.test_embeddings.cpu().numpy())
        np.save(os.path.join(self.opt.save_dir,'train_embeddings'),self.train_embeddings.cpu().numpy())
        np.save(os.path.join(self.opt.save_dir,'test_labels'),self.test_labels.cpu().numpy())
        np.save(os.path.join(self.opt.save_dir,'train_labels'),self.train_labels.cpu().numpy())

        self.epoch += 1

    # def domain_random_loader_init(self,signals,labels,sequences,queue):
    #     pre_thread_iter = self.epoch_iter_length//self.load_thread
    #     last_thread_iter = self.epoch_iter_length - pre_thread_iter*(self.load_thread-1)
    #     for i in range(self.load_thread):
    #         if i != self.load_thread-1:
    #             self.start_process(signals,labels,np.random.choice(sequences,pre_thread_iter*self.opt.batchsize,replace=True),queue)
    #         else:
    #             self.start_process(signals,labels,np.random.choice(sequences,last_thread_iter*self.opt.batchsize,replace=True),queue)

    # def dann_train(self,signals,labels,src_sequences,dst_sequences,beta=1.0,stable=False,plot=True):
    #     self.epoch_forward_init(signals,labels,src_sequences,True)
    #     domains = np.load(os.path.join(self.opt.dataset_dir,'domains.npy'))
    #     domains = dataloader.rebuild_domain(domains)
    #     loss_show = [0,0,0]

    #     if not self.opt.no_dst_domain:
    #         dst_domain_queue = Queue(self.opt.load_thread)
    #         self.domain_random_loader_init(signals,labels,dst_sequences,dst_domain_queue)
            
    #     for i in range(self.epoch_iter_length):
    #         self.step = float(i/self.epoch_iter_length + self.epoch)
    #         p = self.step / self.n_epochs
    #         if stable:alpha = beta
    #         else:alpha = beta * (2. / (1. + np.exp(-10 * p)) - 1)
    #         self.optimizer.zero_grad()
    #         # src
    #         s_signal,s_label,sequence = self.queue.get()
    #         this_batch_len = s_signal.shape[0]
    #         s_signal,s_label = transforms.ToTensor(s_signal,s_label,gpu_id =self.opt.gpu_id)
    #         if self.opt.dann_domain_num == 2:
    #             s_domain = transforms.ToTensor(None,np.zeros(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
    #         else:
    #             s_domain = transforms.batch_generator(None, domains, sequence)
    #             s_domain = transforms.ToTensor(None,s_domain,gpu_id =self.opt.gpu_id)

    #         class_output, domain_output = self.net(s_signal, alpha=alpha)
    #         self.add_label_to_confusion_mat(s_label,class_output,False)
    #         loss_s_label = self.loss_dann_c(class_output, s_label)
    #         loss_s_domain = self.loss_dann_d(domain_output, s_domain)
    #         loss_show[0] += loss_s_label.item()
    #         loss = loss_s_label
    #         if not self.opt.finetune:
    #             loss_show[1] += loss_s_domain.item()
    #             loss = loss+loss_s_domain
            
    #         # dst
    #         if not self.opt.no_dst_domain:
    #             d_signal,_,sequence = dst_domain_queue.get()
    #             d_signal = transforms.ToTensor(d_signal[:this_batch_len],None,gpu_id =self.opt.gpu_id)
    #             if self.opt.dann_domain_num == 2:
    #                 d_domain = transforms.ToTensor(None,np.ones(this_batch_len, dtype=np.int64),gpu_id =self.opt.gpu_id)
    #             else:          
    #                 d_domain = transforms.batch_generator(None, domains, sequence[:this_batch_len])
    #                 d_domain = transforms.ToTensor(None,d_domain,gpu_id=self.opt.gpu_id)
           
    #             _, domain_output = self.net(d_signal, alpha=alpha)
    #             loss_d_domain = self.loss_dann_d(domain_output, d_domain)
    #             loss_show[2] += loss_d_domain.item()

    #             loss = loss +loss_d_domain
         
    #         loss.backward()
    #         self.optimizer.step()
    #     if plot:
    #         self.add_class_acc_to_tensorboard('train')
    #         self.opt.TBGlobalWriter.add_scalars('fold'+str(self.fold+1)+'/loss', {'src_label':loss_show[0]/(i+1),
    #                                                             'src_domain':loss_show[1]/(i+1),
    #                                                             'dst_domain':loss_show[2]/(i+1)}, self.step)
                                    
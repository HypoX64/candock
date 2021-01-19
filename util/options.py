import argparse
import os
import time
import numpy as np
from tensorboardX import SummaryWriter
from . import util,dsp,plot

import sys
sys.path.append("..")
from data import statistics,augmenter,dataloader

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # ------------------------Base------------------------
        self.parser.add_argument('--gpu_id', type=str, default='0',help='choose which gpu want to use, Single GPU: 0 | 1 | 2 ; Multi-GPU: 0,1,2,3 ; No GPU: -1')        
        self.parser.add_argument('--no_cudnn', action='store_true', help='if specified, do not use cudnn')
        self.parser.add_argument('--label', type=str, default='auto',help='number of labels')
        self.parser.add_argument('--input_nc', type=str, default='auto', help='number of input channels')
        self.parser.add_argument('--loadsize', type=str, default='auto', help='load data in this size')
        self.parser.add_argument('--finesize', type=str, default='auto', help='crop your data into this size')
        self.parser.add_argument('--label_name', type=str, default='auto',help='name of labels,example:"a,b,c,d,e,f"')
        self.parser.add_argument('--mode', type=str, default='auto',help='classify_1d | classify_2d | autoencoder | domain')
        self.parser.add_argument('--domain_num', type=str, default='2',
            help='number of domain, only available when mode==domain. 2 | auto ,if input 2, train-data is domain 0,test-data is domain 1.')
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/simple_test',help='your dataset path')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints/',help='save checkpoints')
        self.parser.add_argument('--tensorboard', type=str, default='./checkpoints/tensorboardX',help='tensorboardX log dir')
        self.parser.add_argument('--TBGlobalWriter', type=str, default='',help='')
          
        # ------------------------Training Matters------------------------
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--lr', type=float, default=0.001,help='learning rate') 
        self.parser.add_argument('--batchsize', type=int, default=64,help='batchsize')
        self.parser.add_argument('--load_thread', type=int, default=8,help='how many threads when load data')
        self.parser.add_argument('--best_index', type=str, default='f1',help='select which evaluation index to get the best results in all epochs, f1 | err')
        self.parser.add_argument('--pretrained', type=str, default='',help='pretrained model path. If not specified, fo not use pretrained model')
        self.parser.add_argument('--continue_train', action='store_true', help='if specified, continue train')
        self.parser.add_argument('--weight_mod', type=str, default='auto',help='Choose weight mode: auto | normal')
        self.parser.add_argument('--network_save_freq', type=int, default=5,help='the freq to save network')

        # ------------------------Preprocessing------------------------
        self.parser.add_argument('--normliaze', type=str, default='None', help='mode of normliaze, z-score | 5_95 | maxmin | None')      
        # filter
        self.parser.add_argument('--filter', type=str, default='None', help='type of filter, fft | fir | iir |None')
        self.parser.add_argument('--filter_mod', type=str, default='bandpass', help='mode of fft_filter, bandpass | bandstop')
        self.parser.add_argument('--filter_fs', type=int, default=1000, help='fs of filter')
        self.parser.add_argument('--filter_fc', type=str, default='[]', help='fc of filter, eg. [0.1,10]')

        # filter by wavelet
        self.parser.add_argument('--wave', type=str, default='None', help='wavelet name string, wavelet(eg. dbN symN haar gaus mexh) | None')
        self.parser.add_argument('--wave_level', type=int, default=5, help='decomposition level')
        self.parser.add_argument('--wave_usedcoeffs', type=str, default='[]', help='Coeff used for reconstruction, \
            eg. when level = 6 usedcoeffs=[1,1,0,0,0,0,0] : reconstruct signal with cA6, cD6')
        self.parser.add_argument('--wave_channel', action='store_true', help='if specified, input reconstruct each coeff as a channel.')
        
        
        # ------------------------Data Augmentation------------------------
        # base
        self.parser.add_argument('--augment', type=str, default='None', 
            help='all | scale,warp,app,aaft,iaaft,filp,spike,step,slope,white,pink,blue,brown,violet ,enter some of them')
        self.parser.add_argument('--augment_noise_lambda', type=float, default = 1.0, help='noise level(spike,step,slope,white,pink,blue,brown,violet)')

        # for gan,it only support when fold_index = 1 or 0
        self.parser.add_argument('--gan', action='store_true', help='if specified, using gan to augmente dataset')
        self.parser.add_argument('--gan_lr', type=float, default=0.0002,help='learning rate')
        self.parser.add_argument('--gan_augment_times', type=float, default=2,help='how many times that will be augmented by dcgan')
        self.parser.add_argument('--gan_latent_dim', type=int, default=100,help='dimensionality of the latent space')
        self.parser.add_argument('--gan_labels', type=str, default='[]',help='which label that will be augmented by dcgan, eg: [0,1,2,3]')
        self.parser.add_argument('--gan_epochs', type=int, default=100,help='number of epochs of gan training')

        # ------------------------Dataset------------------------
        """--fold_index
        When --k_fold != 0 or 1:
        Cut dataset into sub-set using index , and then run k-fold with sub-set
        If input 'auto', it will shuffle dataset and then cut dataset to sub-dataset equally
        If input 'load', load indexs.npy as fold_index
        If input: [2,4,6,7]
        when len(dataset) == 10
        sub-set: dataset[0:2],dataset[2:4],dataset[4:6],dataset[6:7],dataset[7:]
        -------
        When --k_fold == 0 or 1:
        If input 'auto', it will shuffle dataset and then cut 80% dataset to train and other to eval
        If input 'load', load indexs.npy as fold_index
        If input: [5]
        when len(dataset) == 10
        train-set : dataset[0:5]  eval-set : dataset[5:]
        """
        self.parser.add_argument('--k_fold', type=int, default=0,help='fold_num of k-fold.If 0 or 1, no k-fold and cut 80% to train and other to eval')
        self.parser.add_argument('--fold_index', type=str, default='auto',
            help='auto | load | "input_your_index"-where to fold, eg. when 5-fold and input: [2,4,6,7] -> sub-set: dataset[0:2],dataset[2:4],dataset[4:6],dataset[6:7],dataset[7:]')
        self.parser.add_argument('--mergelabel', type=str, default='None',
            help='merge some labels to one label and give the result, example:"[[0,1,4],[2,3,5]]" -> label(0,1,4) regard as 0,label(2,3,5) regard as 1')
        self.parser.add_argument('--mergelabel_name', type=str, default='None',help='name of labels,example:"a,b,c,d,e,f"')
        
        # ------------------------Network------------------------
        """Available Network
        1d: lstm, cnn_1d, resnet18_1d, resnet34_1d, multi_scale_resnet_1d,
            micro_multi_scale_resnet_1d,autoencoder,mlp
        2d: mobilenet, dfcnn, multi_scale_resnet, resnet18, resnet50, resnet101,
            densenet121, densenet201, squeezenet
        """
        self.parser.add_argument('--model_name', type=str, default='micro_multi_scale_resnet_1d',help='Choose model  lstm...')
        self.parser.add_argument('--lstm_inputsize', type=str, default='auto',help='lstm_inputsize of LSTM')
        self.parser.add_argument('--lstm_timestep', type=int, default=100,help='time_step of LSTM')
        # For autoecoder
        self.parser.add_argument('--feature', type=int, default=3, help='number of encoder features')
        # For 2d network(stft spectrum)
        # Please cheek ./save_dir/spectrum_eg.jpg to change the following parameters
        self.parser.add_argument('--spectrum', type=str, default='stft', help='stft | cwt')
        self.parser.add_argument('--spectrum_n_downsample', type=int, default=1, help='downsample befor convert to spectrum')
        self.parser.add_argument('--cwt_wavename', type=str, default='cgau8', help='')
        self.parser.add_argument('--cwt_scale_num', type=int, default=64, help='')
        self.parser.add_argument('--stft_size', type=int, default=512, help='length of each fft segment')
        self.parser.add_argument('--stft_stride', type=int, default=128, help='stride of each fft segment')
        self.parser.add_argument('--stft_no_log', action='store_true', help='if specified, do not log1p spectrum')
        self.parser.add_argument('--img_shape', type=str, default='auto', help='output shape of stft. It depend on \
            stft_size,stft_stride,stft_n_downsample. Do not input this parameter.')

        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.gpu_id != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.opt.gpu_id)

        if self.opt.label != 'auto':
            self.opt.label = int(self.opt.label)
        if self.opt.input_nc !='auto':
            self.opt.input_nc = int(self.opt.input_nc)
        if self.opt.loadsize !='auto':
            self.opt.loadsize = int(self.opt.loadsize)
        if self.opt.finesize !='auto':
            self.opt.finesize = int(self.opt.finesize)
        if self.opt.lstm_inputsize != 'auto':
            self.opt.lstm_inputsize = int(self.opt.lstm_inputsize)

        if self.opt.mode == 'auto':
            if self.opt.model_name in ['lstm', 'cnn_1d', 'resnet18_1d', 'resnet34_1d', 
                'multi_scale_resnet_1d','micro_multi_scale_resnet_1d','mlp']:
                self.opt.mode = 'classify_1d'
            elif self.opt.model_name in ['light','dfcnn', 'multi_scale_resnet', 'resnet18', 'resnet50',
                'resnet101','densenet121', 'densenet201', 'squeezenet', 'mobilenet','EarID','MV_Emotion']:
                self.opt.mode = 'classify_2d'
            elif self.opt.model_name == 'autoencoder':
                self.opt.mode = 'autoencoder'
            elif self.opt.model_name in ['dann','dann_base']:
                self.opt.mode = 'domain'
            else:
                print('\033[1;31m'+'Error: do not support this network '+self.opt.model_name+'\033[0m')
                sys.exit(0)

        if self.opt.k_fold == 0 :
            self.opt.k_fold = 1

        if self.opt.fold_index == 'auto':
            if os.path.isfile(os.path.join(self.opt.dataset_dir,'index.npy')):
                print('Warning: index.npy exists but does not load it')
        elif self.opt.fold_index == 'load':
            if os.path.isfile(os.path.join(self.opt.dataset_dir,'index.npy')):
                self.opt.fold_index = (np.load(os.path.join(self.opt.dataset_dir,'index.npy'))).tolist()
            else:
                print('Warning: index.npy does not exist')
                sys.exit(0)
        else:
            self.opt.fold_index = eval(self.opt.fold_index)


        if self.opt.augment == 'all':
            self.opt.augment = ['scale','warp','spike','step','slope','white','pink','blue','brown','violet','app','aaft','iaaft','filp']
        else:
            self.opt.augment = str2list(self.opt.augment)

        self.opt.filter_fc = eval(self.opt.filter_fc)
        self.opt.wave_usedcoeffs = eval(self.opt.wave_usedcoeffs)
        self.opt.gan_labels = eval(self.opt.gan_labels)

        self.opt.mergelabel = eval(self.opt.mergelabel)
        if self.opt.mergelabel_name != 'None':
            self.opt.mergelabel_name = self.opt.mergelabel_name.replace(" ", "").split(",")

        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        localtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        util.makedirs(self.opt.save_dir)
        util.writelog(str(localtime)+'\n'+message+'\n', self.opt,True)

        # start tensorboard
        self.opt.tensorboard = os.path.join(self.opt.tensorboard,localtime+'_'+os.path.split(self.opt.save_dir)[1])
        self.opt.TBGlobalWriter = SummaryWriter(self.opt.tensorboard)
        util.writelog('Please run "tensorboard --logdir checkpoints/tensorboardX --host=your_server_ip" and input "'+localtime+'" to filter outputs',self.opt,True)
        self.opt.TBGlobalWriter.add_text('Opt', message.replace('\n', '  \n'))
        
        return self.opt

def str2list(string,out_type = 'string'):
    out_list = []
    string = string.replace(' ','').replace('[','').replace(']','')
    strings = string.split(',')
    for string in strings:
        if string != '':
            if out_type == 'string':
                out_list.append(string)
            elif out_type == 'int':
                out_list.append(int(string))
            elif out_type == 'float':
                out_list.append(float(string))
    return out_list

def get_auto_options(opt,signals,labels):
    
    shape = signals.shape
    label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
    if opt.label =='auto':
        opt.label = label_num
    if opt.input_nc =='auto':
        opt.input_nc = shape[1]
    if opt.loadsize =='auto':
        opt.loadsize = shape[2]
    if opt.finesize =='auto':
        opt.finesize = int(shape[2]*0.9)
    if opt.lstm_inputsize =='auto':
        opt.lstm_inputsize = opt.finesize//opt.lstm_timestep

    # weight
    opt.weight = np.ones(opt.label)
    if opt.weight_mod == 'auto':
        opt.weight = 1/label_cnt_per
        opt.weight = opt.weight/np.min(opt.weight)
        util.writelog('Loss_weight:'+str(opt.weight),opt,True,True)
    import torch
    opt.weight = torch.from_numpy(opt.weight).float()
    if opt.gpu_id != '-1':      
        opt.weight = opt.weight.cuda()

    # label name
    if opt.label_name == 'auto':
        names = []
        for i in range(opt.label):
            names.append(str(i))
        opt.label_name = names
    elif not isinstance(opt.label_name,list):
        opt.label_name = opt.label_name.replace(" ", "").split(",")

    # domain_num
    if opt.mode == 'domain':
        if opt.domain_num == '2':
            opt.domain_num = 2
        else:
            if os.path.isfile(os.path.join(opt.dataset_dir,'domains.npy')):
                domains = np.load(os.path.join(opt.dataset_dir,'domains.npy'))
                domains = dataloader.rebuild_domain(domains)
                opt.domain_num = statistics.label_statistics(domains)[2]
            else:
                print('Please generate domains.npy(np.int64, shape like labels.npy)')
                sys.exit(0)

    # check stft spectrum
    if opt.mode in ['classify_2d','domain'] and signals.ndim == 3:
        spectrums = []
        data = signals[np.random.randint(0,shape[0]-1)].reshape(1,shape[1],shape[2])
        data = augmenter.base1d(opt, data, test_flag=False)[0]
        plot.draw_eg_signals(data,opt)
        for i in range(shape[1]):
            spectrums.append(dsp.signal2spectrum(data[i],opt.stft_size,opt.stft_stride,
                opt.cwt_wavename,opt.cwt_scale_num,opt.spectrum_n_downsample,not opt.stft_no_log, mod = opt.spectrum))
        plot.draw_eg_spectrums(spectrums,opt)
        opt.img_shape = spectrums[0].shape
        h,w = opt.img_shape
        print('Shape of stft spectrum h,w:',opt.img_shape)
        print('\033[1;37m'+'Please cheek tensorboard->IMAGES->spectrum_eg to change parameters'+'\033[0m')
        
        if h<64 or w<64:
            print('\033[1;33m'+'Warning: spectrum is too small'+'\033[0m') 
        if h>512 or w>512:
            print('\033[1;33m'+'Warning: spectrum is too large'+'\033[0m')
    
    if signals.ndim == 4:
        opt.img_shape = signals.shape[2],signals.shape[3]
        img = signals[np.random.randint(0,shape[0]-1)]
        opt.TBGlobalWriter.add_image('img_eg',img)

    return opt
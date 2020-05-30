import argparse
import os
import time
import numpy as np
from . import util,dsp,plot

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # ------------------------Base------------------------
        self.parser.add_argument('--gpu_id', type=int, default=0,help='choose which gpu want to use, 0 | 1 | 2 ...')        
        self.parser.add_argument('--no_cudnn', action='store_true', help='if specified, do not use cudnn')
        self.parser.add_argument('--label', type=str, default='auto',help='number of labels')
        self.parser.add_argument('--input_nc', type=str, default='auto', help='of input channels')
        self.parser.add_argument('--loadsize', type=str, default='auto', help='load data in this size')
        self.parser.add_argument('--finesize', type=str, default='auto', help='crop your data into this size')
        self.parser.add_argument('--label_name', type=str, default='auto',help='name of labels,example:"a,b,c,d,e,f"')
        
        # ------------------------Dataset------------------------
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/simple_test',help='your dataset path')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints/',help='save checkpoints')
        self.parser.add_argument('--separated', action='store_true', help='if specified,for preload data, if input, load separated train and test datasets')
        self.parser.add_argument('--no_shuffle', action='store_true', help='if specified, do not shuffle data when load(use to evaluate individual differences)')
        self.parser.add_argument('--load_thread', type=int, default=8,help='how many threads when load data')  
        self.parser.add_argument('--normliaze', type=str, default='5_95', help='mode of normliaze, 5_95 | maxmin | None')      

        # ------------------------Network------------------------
        """Available Network
        1d: lstm, cnn_1d, resnet18_1d, resnet34_1d, multi_scale_resnet_1d,
            micro_multi_scale_resnet_1d,autoencoder
        2d: mobilenet, dfcnn, multi_scale_resnet, resnet18, resnet50, resnet101,
            densenet121, densenet201, squeezenet
        """
        self.parser.add_argument('--model_name', type=str, default='micro_multi_scale_resnet_1d',help='Choose model  lstm...')
        self.parser.add_argument('--model_type', type=str, default='auto',help='1d | 2d')
        # For lstm 
        self.parser.add_argument('--lstm_inputsize', type=str, default='auto',help='lstm_inputsize of LSTM')
        self.parser.add_argument('--lstm_timestep', type=int, default=100,help='time_step of LSTM')
        # For autoecoder
        self.parser.add_argument('--feature', type=int, default=3, help='number of encoder features')
        # For 2d network(stft spectrum)
        # Please cheek ./save_dir/spectrum_eg.jpg to change the following parameters
        self.parser.add_argument('--stft_size', type=int, default=512, help='length of each fft segment')
        self.parser.add_argument('--stft_stride', type=int, default=128, help='stride of each fft segment')
        self.parser.add_argument('--stft_n_downsample', type=int, default=1, help='downsample befor stft')
        self.parser.add_argument('--stft_no_log', action='store_true', help='if specified, do not log1p spectrum')
        self.parser.add_argument('--stft_shape', type=str, default='auto', help='shape of stft. It depend on \
            stft_size,stft_stride,stft_n_downsample. Do not input this parameter.')

        # ------------------------Training Matters------------------------
        self.parser.add_argument('--pretrained', type=str, default='',help='pretrained model path. If not specified, fo not use pretrained model')
        self.parser.add_argument('--continue_train', action='store_true', help='if specified, continue train')
        self.parser.add_argument('--lr', type=float, default=0.001,help='learning rate') 
        self.parser.add_argument('--batchsize', type=int, default=64,help='batchsize')
        self.parser.add_argument('--weight_mod', type=str, default='auto',help='Choose weight mode: auto | normal')
        self.parser.add_argument('--epochs', type=int, default=20,help='end epoch')
        self.parser.add_argument('--network_save_freq', type=int, default=5,help='the freq to save network')
        self.parser.add_argument('--k_fold', type=int, default=0,help='fold_num of k-fold.if 0 or 1,no k-fold')
        self.parser.add_argument('--mergelabel', type=str, default='None',
            help='merge some labels to one label and give the result, example:"[[0,1,4],[2,3,5]]" -> label(0,1,4) regard as 0,label(2,3,5) regard as 1')
        self.parser.add_argument('--mergelabel_name', type=str, default='None',help='name of labels,example:"a,b,c,d,e,f"')
        
        self.initialized = True

    def getparse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.gpu_id != -1:
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

        if self.opt.model_type == 'auto':
            if self.opt.model_name in ['lstm', 'cnn_1d', 'resnet18_1d', 'resnet34_1d', 
                'multi_scale_resnet_1d','micro_multi_scale_resnet_1d','autoencoder']:
                self.opt.model_type = '1d'
            elif self.opt.model_name in ['dfcnn', 'multi_scale_resnet', 'resnet18', 'resnet50',
                'resnet101','densenet121', 'densenet201', 'squeezenet', 'mobilenet']:
                self.opt.model_type = '2d'
            else:
                print('\033[1;31m'+'Error: do not support this network '+self.opt.model_name+'\033[0m')
                exit(0)

        if self.opt.k_fold == 0 :
            self.opt.k_fold = 1

        if self.opt.separated:
            self.opt.k_fold = 1

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
        localtime = time.asctime(time.localtime(time.time()))
        util.makedirs(self.opt.save_dir)
        util.writelog(str(localtime)+'\n'+message, self.opt,True)

        return self.opt

def get_auto_options(opt,label_cnt_per,label_num,signals):
    
    shape = signals.shape
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
    util.writelog('Loss_weight:'+str(opt.weight),opt,True)
    import torch
    opt.weight = torch.from_numpy(opt.weight).float()
    if opt.gpu_id != -1:      
        opt.weight = opt.weight.cuda()

    # label name
    if opt.label_name == 'auto':
        names = []
        for i in range(opt.label):
            names.append(str(i))
        opt.label_name = names
    elif not isinstance(opt.label_name,list):
        opt.label_name = opt.label_name.replace(" ", "").split(",")


    # check stft spectrum
    if opt.model_type =='2d':
        spectrums = []
        data = signals[np.random.randint(0,shape[0]-1)]
        for i in range(shape[1]):
            spectrums.append(dsp.signal2spectrum(data[i],opt.stft_size, opt.stft_stride, opt.stft_n_downsample, not opt.stft_no_log))
        plot.draw_spectrums(spectrums,opt)
        opt.stft_shape = spectrums[0].shape
        h,w = opt.stft_shape
        print('Shape of stft spectrum h,w:',opt.stft_shape)
        print('\033[1;37m'+'Please cheek ./save_dir/spectrum_eg.jpg to change parameters'+'\033[0m')
        
        if h<64 or w<64:
            print('\033[1;33m'+'Warning: spectrum is too small'+'\033[0m') 
        if h>512 or w>512:
            print('\033[1;33m'+'Warning: spectrum is too large'+'\033[0m')

    return opt
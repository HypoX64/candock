import argparse
import os
import time
import numpy as np
from . import util,dsp,plot,statistics

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
        
        # ------------------------Preprocessing------------------------
        self.parser.add_argument('--normliaze', type=str, default='5_95', help='mode of normliaze, 5_95 | maxmin | None')      
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
        self.parser.add_argument('--augment', type=str, default='all', help='all | scale,filp,amp,noise | scale,filp ....')
        # fft channel --> use fft to improve frequency domain information.
        self.parser.add_argument('--augment_fft', action='store_true', help='if specified, use fft to improve frequency domain informationa')

        # for gan,it only support when fold_index = 1 or 0 now
        # only support when k_fold =0 or 1
        self.parser.add_argument('--gan', action='store_true', help='if specified, using gan to augmente dataset')
        self.parser.add_argument('--gan_lr', type=float, default=0.0002,help='learning rate')
        self.parser.add_argument('--gan_augment_times', type=float, default=2,help='how many times that will be augmented by dcgan')
        self.parser.add_argument('--gan_latent_dim', type=int, default=100,help='dimensionality of the latent space')
        self.parser.add_argument('--gan_labels', type=str, default='[]',help='which label that will be augmented by dcgan, eg: [0,1,2,3]')
        self.parser.add_argument('--gan_epochs', type=int, default=100,help='number of epochs of gan training')

        # ------------------------Dataset------------------------
        """--fold_index
        5-fold:
        Cut dataset into sub-set using index , and then run k-fold with sub-set
        If input 'auto', it will shuffle dataset and then cut dataset equally
        If input: [2,4,6,7]
        when len(dataset) == 10
        sub-set: dataset[0:2],dataset[2:4],dataset[4:6],dataset[6:7],dataset[7:]
        -------
        No-fold:
        If input 'auto', it will shuffle dataset and then cut 80% dataset to train and other to eval
        If input: [5]
        when len(dataset) == 10
        train-set : dataset[0:5]  eval-set : dataset[5:]
        """
        self.parser.add_argument('--fold_index', type=str, default='auto',
            help='where to fold, eg. when 5-fold and input: [2,4,6,7] -> sub-set: dataset[0:2],dataset[2:4],dataset[4:6],dataset[6:7],dataset[7:]')
        self.parser.add_argument('--k_fold', type=int, default=0,help='fold_num of k-fold.If 0 or 1, no k-fold and cut 0.8 to train and other to eval')
        self.parser.add_argument('--dataset_dir', type=str, default='./datasets/simple_test',help='your dataset path')
        self.parser.add_argument('--save_dir', type=str, default='./checkpoints/',help='save checkpoints')
        self.parser.add_argument('--load_thread', type=int, default=8,help='how many threads when load data')  
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
                'multi_scale_resnet_1d','micro_multi_scale_resnet_1d','autoencoder','mlp']:
                self.opt.model_type = '1d'
            elif self.opt.model_name in ['dfcnn', 'multi_scale_resnet', 'resnet18', 'resnet50',
                'resnet101','densenet121', 'densenet201', 'squeezenet', 'mobilenet']:
                self.opt.model_type = '2d'
            else:
                print('\033[1;31m'+'Error: do not support this network '+self.opt.model_name+'\033[0m')
                exit(0)

        if self.opt.k_fold == 0 :
            self.opt.k_fold = 1

        if self.opt.fold_index != 'auto':
            self.opt.fold_index = eval(self.opt.fold_index)

        if os.path.isfile(os.path.join(self.opt.dataset_dir,'index.npy')):
            self.opt.fold_index = (np.load(os.path.join(self.opt.dataset_dir,'index.npy'))).tolist()

        if self.opt.augment == 'all':
            self.opt.augment = ["scale","filp","amp","noise"]
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
        localtime = time.asctime(time.localtime(time.time()))
        util.makedirs(self.opt.save_dir)
        util.writelog(str(localtime)+'\n'+message, self.opt,True)

        return self.opt

def str2list(string,out_type = 'string'):
    out_list = []
    string = string.replace(' ','').replace('[','').replace(']','')
    strings = string.split(',')
    for string in strings:
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
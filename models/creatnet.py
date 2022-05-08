import sys
import torch
from torch import nn
from .net_1d import cnn_1d,lstm,resnet_1d,multi_scale_resnet_1d,micro_multi_scale_resnet_1d,mlp
from .net_2d import dfcnn,lightcnn,densenet,ghostnet,mnasnet,mobilenet,resnet,resnet_cbam,shufflenetv2,squeezenet
from .domain import dann,dann_lstm
from .ipmc import EarID
from . import model_util
from . dml import dml


def creatnet(opt):
    name = opt.model_name
    #---------------------------------classify_1d---------------------------------
    #mlp
    if name =='mlp':
        net = mlp.mlp(opt.input_nc, opt.label, opt.finesize)
    #lstm
    elif name =='lstm':
        net =  lstm.lstm(opt.lstm_inputsize,opt.lstm_timestep,input_nc=opt.input_nc,num_classes=opt.label)
    #cnn
    elif name == 'cnn_1d':
        net = cnn_1d.cnn(opt.input_nc,num_classes=opt.label)
    elif name == 'resnet18_1d':
        net = resnet_1d.resnet18()
        net.conv1 = nn.Conv1d(opt.input_nc, 64, 7, 2, 3, bias=False)
        net.fc = nn.Linear(512, opt.label)
    elif name == 'resnet34_1d':
        net = resnet_1d.resnet34()
        net.conv1 = nn.Conv1d(opt.input_nc, 64, 7, 2, 3, bias=False)
        net.fc = nn.Linear(512, opt.label)
    elif name == 'multi_scale_resnet_1d':
        net = multi_scale_resnet_1d.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=opt.label)
    elif name == 'micro_multi_scale_resnet_1d':
        net = micro_multi_scale_resnet_1d.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=opt.label)

    #---------------------------------classify_2d---------------------------------
    elif name == 'light':
        net = lightcnn.LightCNN(input_nc=opt.input_nc, num_classes=opt.label)
    elif name == 'dfcnn':
        net = dfcnn.dfcnn(num_classes = opt.label, input_nc = opt.input_nc)
    
    elif name in ['resnet101','resnet50','resnet18','resnet18_cbam','resnet50_cbam']:
        if name =='resnet101':
            net = resnet.resnet101(pretrained=True)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet50':
            net = resnet.resnet50(pretrained=True)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet18':
            net = resnet.resnet18(pretrained=True)
            net.fc = nn.Linear(512, opt.label)
        elif name == 'resnet18_cbam':
            net = resnet_cbam.resnet18_cbam(pretrained=True)
        elif name == 'resnet50_cbam':
            net = resnet_cbam.resnet50_cbam(pretrained=True)
        net.conv1 = nn.Conv2d(opt.input_nc, 64, 7, 2, 3, bias=False)        
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = densenet.densenet121(pretrained=True,num_classes = opt.label)
        elif name == 'densenet201':
            net = densenet.densenet201(pretrained=True,num_classes = opt.label)
        net.features.conv0 = nn.Conv2d(opt.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False) 

    elif name == 'mobilenetv2':
        net = mobilenet.mobilenet_v2(pretrained=True)
        net.features[0][0] = nn.Conv2d(opt.input_nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[1] = nn.Linear(in_features=1280, out_features=opt.label, bias=True)

    elif name == 'mobilenetv3':
        net = mobilenet.mobilenet_v3_large()
        net.features[0][0] = nn.Conv2d(opt.input_nc, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        net.classifier[3] = nn.Linear(in_features=1280, out_features=opt.label, bias=True)

    elif name == 'ghostnet':
        net = ghostnet.ghost_net(num_classes=opt.label)
        net.features[0][0] = nn.Conv2d(opt.input_nc, 16, 3, 2, 1, bias=False)

    elif name == 'shufflenetv2':
        net = shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        net.conv1[0]=nn.Conv2d(opt.input_nc, 24, 3, 2, 1, bias=False)
        net.fc = nn.Linear(1024, opt.label)
    
    elif name == 'mnasnet':
        net = mnasnet.mnasnet1_0(pretrained=True)
        net.layers[0] = nn.Conv2d(opt.input_nc, 32, 3, padding=1, stride=2, bias=False)
        net.classifier[1] = nn.Linear(1280, opt.label)

    elif name == 'squeezenet':
        net = squeezenet.squeezenet1_1(pretrained=True)
        net.features[0] = nn.Conv2d(opt.input_nc, 64, kernel_size=3, stride=2)
        net.classifier[1] = nn.Conv2d(512, opt.label, kernel_size=1)

    #---------------------------------Deep Metric Learning---------------------------------
    if opt.dml:
        if opt.model_name == 'earid':
            net = EarID.EarID(opt.n_embedding)
        else:
            feature_nums = {'resnet18':512,
                            'resnet50':2048,
                            'resnet18_cbam':512,
                            'resnet50_cbam':2048,
                            'mobilenetv2': 1280,
                            'mobilenetv3':960,
                            'ghostnet':960,
                            'mnasnet':1280,
                            'squeezenet':512,
                            'shufflenetv2':1024}
            if opt.model_name in ['resnet18','resnet50','mobilenetv3','resnet18_cbam','resnet50_cbam']:
                encoder = torch.nn.Sequential(*(list(net.children())[:-2]))
            else: 
                encoder = torch.nn.Sequential(*(list(net.children())[:-1]))
            net = dml.DML(encoder,feature_nums[opt.model_name],opt.n_embedding)
        

    #---------------------------------domain---------------------------------
    if opt.dann:
        avg_pool_flag = True
        # only supported： lstm,resnet18,mobilenetv2,mobilenetv3,ghostnet,mnasnet,squeezenet,shufflenetv2
        if opt.model_name in ['resnet18','mobilenetv3']:
            encoder = torch.nn.Sequential(*(list(net.children())[:-2]))
        else: 
            encoder = torch.nn.Sequential(*(list(net.children())[:-1]))
        
        supports =         ['lstm',  'resnet18','mobilenetv2','mobilenetv3','ghostnet','mnasnet','squeezenet','shufflenetv2']
        if avg_pool_flag:
            feature_nums = [128*opt.label,512,    1280,         960,          960,      1280,      512,         1024 ]
        else:
            # only for input size 257*251
            feature_nums = [128*opt.label,512*9*8,1280*9*8,     960*9*8,      960*9*8,  1280*9*8,  512*15*16,   1024*9*8 ]
        if opt.model_name == 'lstm':
            net = dann_lstm.DANN(opt.lstm_inputsize, opt.lstm_timestep, opt.input_nc, opt.label, opt.dann_domain_num)
        else:        
            net = dann.DANN(encoder,opt.label,opt.dann_domain_num,feature_nums[supports.index(opt.model_name)],avg_pool=avg_pool_flag)

    #---------------------------------finetune---------------------------------
    if opt.finetune:
        if opt.dann:
            model_util.freeze_besides_names(net, ['class_classifier','domain_classifier'])
        else:
            if name in ['lstm','resnet18','resnet50','resnet101','shufflenetv2']:
                model_util.freeze_besides_names(net, 'fc')

            elif name in ['lstm','densenet121','densenet201','mobilenet','mobilenetv2','mobilenetv3','ghostnet','mnasnet','squeezenet']:
                model_util.freeze_besides_names(net, ['classifier'])
            
    if opt.mode in ['classify_2d','domain']:
        exp = torch.rand(opt.batchsize, opt.input_nc, opt.img_shape[0], opt.img_shape[1])
    else:
        exp = torch.rand(opt.batchsize,opt.input_nc,opt.finesize)

    return net,exp

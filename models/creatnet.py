from torch import nn
from .net_1d import cnn_1d,lstm,resnet_1d,multi_scale_resnet_1d,micro_multi_scale_resnet_1d,autoencoder
from .net_2d import densenet,dfcnn,mobilenet,resnet,squeezenet,multi_scale_resnet


def creatnet(opt):
    name = opt.model_name
    #---------------------------------1d---------------------------------
    #encoder
    if name =='autoencoder':
        net = autoencoder.Autoencoder(opt.input_nc, opt.feature, opt.label,opt.finesize)
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

    #---------------------------------2d---------------------------------
    elif name == 'dfcnn':
        net = dfcnn.dfcnn(num_classes = opt.label)
    elif name == 'multi_scale_resnet':
        net = multi_scale_resnet.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=opt.label)
    elif name in ['resnet101','resnet50','resnet18']:
        if name =='resnet101':
            net = resnet.resnet101(pretrained=True)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet50':
            net = resnet.resnet50(pretrained=True)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet18':
            net = resnet.resnet18(pretrained=True)
            net.fc = nn.Linear(512, opt.label)
        net.conv1 = nn.Conv2d(opt.input_nc, 64, 7, 2, 3, bias=False)        
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = densenet.densenet121(pretrained=True,num_classes=opt.label)
        elif name == 'densenet201':
            net = densenet.densenet201(pretrained=True,num_classes=opt.label)
    elif name =='squeezenet':
        net = squeezenet.squeezenet1_1(pretrained=True,num_classes=opt.label,inchannel = 1)

    return net
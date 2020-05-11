from torch import nn
from . import cnn_1d,densenet,dfcnn,lstm,mobilenet,resnet,resnet_1d,squeezenet, \
multi_scale_resnet,multi_scale_resnet_1d,micro_multi_scale_resnet_1d,autoencoder
# from models import cnn_1d,densenet,dfcnn,lstm,mobilenet,resnet,resnet_1d,squeezenet
# from models import multi_scale_resnet,multi_scale_resnet_1d,micro_multi_scale_resnet_1d

def CreatNet(opt):
    name = opt.model_name

    #encoder
    if name =='autoencoder':
        net = autoencoder.Autoencoder(opt.input_nc, opt.feature, opt.label,opt.finesize)
    #1d
    elif name =='lstm':
        net =  lstm.lstm(opt.input_size,opt.time_step,input_nc=opt.input_nc,num_classes=opt.label)
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
    elif name == 'multi_scale_resnet':
        net = multi_scale_resnet.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=opt.label)
    #2d
    elif name == 'dfcnn':
        net = dfcnn.dfcnn(num_classes = opt.label)
    elif name in ['resnet101','resnet50','resnet18']:
        if name =='resnet101':
            net = resnet.resnet101(pretrained=False)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet50':
            net = resnet.resnet50(pretrained=False)
            net.fc = nn.Linear(2048, opt.label)
        elif name =='resnet18':
            net = resnet.resnet18(pretrained=False)
            net.fc = nn.Linear(512, opt.label)
        net.conv1 = nn.Conv2d(opt.input_nc, 64, 7, 2, 3, bias=False)        
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = densenet.densenet121(pretrained=False,num_classes=opt.label)
        elif name == 'densenet201':
            net = densenet.densenet201(pretrained=False,num_classes=opt.label)
    elif name =='squeezenet':
        net = squeezenet.squeezenet1_1(pretrained=False,num_classes=opt.label,inchannel = 1)

    return net
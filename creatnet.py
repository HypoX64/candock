from torch import nn
from models import cnn_1d,densenet,dfcnn,lstm,mobilenet,resnet,resnet_1d,squeezenet
from models import multi_scale_resnet,multi_scale_resnet_1d,micro_multi_scale_resnet_1d

def CreatNet(opt):
    name = opt.model_name
    label_num = opt.label
    if name =='lstm':
        net =  lstm.lstm(100,27,num_classes=label_num)
    elif name == 'cnn_1d':
        net = cnn_1d.cnn(opt.input_nc,num_classes=label_num)
    elif name == 'resnet18_1d':
        net = resnet_1d.resnet18()
        net.conv1 = nn.Conv1d(opt.input_nc, 64, 7, 2, 3, bias=False)
        net.fc = nn.Linear(512, label_num)
    elif name == 'multi_scale_resnet_1d':
        net = multi_scale_resnet_1d.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=label_num)
    elif name == 'micro_multi_scale_resnet_1d':
        net = micro_multi_scale_resnet_1d.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=label_num)
    elif name == 'multi_scale_resnet':
        net = multi_scale_resnet.Multi_Scale_ResNet(inchannel=opt.input_nc, num_classes=label_num)
    elif name == 'dfcnn':
        net = dfcnn.dfcnn(num_classes = label_num)
    elif name in ['resnet101','resnet50','resnet18']:
        if name =='resnet101':
            net = resnet.resnet101(pretrained=False)
            net.fc = nn.Linear(2048, label_num)
        elif name =='resnet50':
            net = resnet.resnet50(pretrained=False)
            net.fc = nn.Linear(2048, label_num)
        elif name =='resnet18':
            net = resnet.resnet18(pretrained=False)
            net.fc = nn.Linear(512, label_num)
        net.conv1 = nn.Conv2d(opt.input_nc, 64, 7, 2, 3, bias=False)        
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = densenet.densenet121(pretrained=False,num_classes=label_num)
        elif name == 'densenet201':
            net = densenet.densenet201(pretrained=False,num_classes=label_num)
    elif name =='squeezenet':
        net = squeezenet.squeezenet1_1(pretrained=False,num_classes=label_num,inchannel = 1)

    return net
from torch import nn
from models import cnn_1d,densenet,dfcnn,lstm,mobilenet,resnet,resnet_1d,squeezenet
from models import multi_scale_resnet,multi_scale_resnet_1d

def CreatNet(name):
    if name =='lstm':
        net =  lstm.lstm(100,27,num_classes=5)
    elif name == 'cnn_1d':
        net = cnn_1d.cnn(1,num_classes=5)
    elif name == 'resnet18_1d':
        net = resnet_1d.resnet18()
        net.conv1 = nn.Conv1d(1, 64, 7, 2, 3, bias=False)
        net.fc = nn.Linear(512, 5)
    elif name == 'multi_scale_resnet_1d':
        net = multi_scale_resnet_1d.Multi_Scale_ResNet(inchannel=1, num_classes=5)
    elif name == 'multi_scale_resnet':
        net = multi_scale_resnet.Multi_Scale_ResNet(inchannel=1, num_classes=5)
    elif name == 'dfcnn':
        net = dfcnn.dfcnn(num_classes = 5)
    elif name in ['resnet101','resnet50','resnet18']:
        if name =='resnet101':
            net = resnet.resnet101(pretrained=False)
            net.fc = nn.Linear(2048, 5)
        elif name =='resnet50':
            net = resnet.resnet50(pretrained=False)
            net.fc = nn.Linear(2048, 5)
        elif name =='resnet18':
            net = resnet.resnet18(pretrained=False)
            net.fc = nn.Linear(512, 5)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)        
    
    elif 'densenet' in name:
        if name =='densenet121':
            net = densenet.densenet121(pretrained=False,num_classes=5)
        elif name == 'densenet201':
            net = densenet.densenet201(pretrained=False,num_classes=5)
    elif name =='squeezenet':
        net = squeezenet.squeezenet1_1(pretrained=False,num_classes=5,inchannel = 1)

    return net
import numpy as np
import torch

import os
import sys
from collections.abc import Iterable
sys.path.append("..")
from util import util

def show_paramsnumber(net,opt):
    parameters = sum(param.numel() for param in net.parameters())
    parameters = round(parameters/1e6,2)
    util.writelog('net parameters: '+str(parameters)+'M',opt,True,True)

'''
固定网络权重
作者：有糖吃可好
链接：https://www.zhihu.com/question/311095447/answer/589307812
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

# 冻结第一层
freeze_by_idxs(model, 0)
# 冻结第一、二层
freeze_by_idxs(model, [0, 1])
#冻结倒数第一层
freeze_by_idxs(model, -1)
# 解冻第一层
unfreeze_by_idxs(model, 0)
# 解冻倒数第一层
unfreeze_by_idxs(model, -1)


# 冻结 em层
freeze_by_names(model, 'em')
# 冻结 fc1, fc3层
freeze_by_names(model, ('fc1', 'fc3'))
# 解冻em, fc1, fc3层
unfreeze_by_names(model, ('em', 'fc1', 'fc3'))

'''

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)


############################################################
def set_freeze_besides_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            for param in child.parameters():
                param.requires_grad = not freeze

def freeze_besides_names(model, layer_names):
    set_freeze_besides_names(model, layer_names, True)

def unfreeze_besides_names(model, layer_names):
    set_freeze_besides_names(model, layer_names, False)

def check_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            print(name,param.requires_grad)
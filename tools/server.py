import os
import time
import shutil
import random
import torch
import numpy as np

import sys
sys.path.append("..")
from util import util,transformer,dataloader,statistics,plot,options
from util import array_operation as arr
from models import creatnet,core

# -----------------------------Init-----------------------------
opt = options.Options()
opt.parser.add_argument('--rec_tmp',type=str,default='./server_data/rec_data', help='')
opt = opt.getparse()
opt.k_fold = 0
opt.save_dir = './checkpoints'
util.makedirs(opt.save_dir)
util.makedirs(opt.rec_tmp)

# -----------------------------Load original data-----------------------------
signals,labels = dataloader.loaddataset(opt)
ori_signals_train,ori_labels_train,ori_signals_eval,ori_labels_eval = \
signals[:opt.fold_index[0]].copy(),labels[:opt.fold_index[0]].copy(),signals[opt.fold_index[0]:].copy(),labels[opt.fold_index[0]:].copy()
label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
opt = options.get_auto_options(opt, label_cnt_per, label_num, ori_signals_train)

# -----------------------------def network-----------------------------
core = core.Core(opt)
core.network_init(printflag=True)

# -----------------------------train-----------------------------
def train(opt):
    core.network_init(printflag=True)

    categorys = os.listdir(opt.rec_tmp)
    categorys.sort()
    print('categorys:',categorys)
    category_num = len(categorys)

    received_signals = [];received_labels = []

    sample_num = 1000
    for i in range(category_num):
        samples = os.listdir(os.path.join(opt.rec_tmp,categorys[i]))
        random.shuffle(samples)
        for j in range(len(samples)):
            txt = util.loadtxt(os.path.join(opt.rec_tmp,categorys[i],samples[j]))
            #print(os.path.join('./datasets/server/data',categorys[i],sample))
            txt_split = txt.split()
            signal_ori = np.zeros(len(txt_split))
            for point in range(len(txt_split)):
                signal_ori[point] = float(txt_split[point])

            for x in range(sample_num//len(samples)):
                ran = random.randint(1000, len(signal_ori)-2000-1)
                this_signal = signal_ori[ran:ran+2000]
                this_signal = arr.normliaze(this_signal,'5_95',truncated=4)
                
                received_signals.append(this_signal)
                received_labels.append(i)

    received_signals = np.array(received_signals).reshape(-1,opt.input_nc,opt.loadsize)
    received_labels = np.array(received_labels).reshape(-1,1)
    received_signals_train,received_labels_train,received_signals_eval,received_labels_eval=\
    dataloader.segment_dataset(received_signals, received_labels, 0.8,random=False)
    print(received_signals_train.shape,received_signals_eval.shape)

    '''merge data'''
    signals_train,labels_train = dataloader.del_labels(ori_signals_train,ori_labels_train, np.linspace(0, category_num-1,category_num,dtype=np.int64))
    signals_eval,labels_eval = dataloader.del_labels(ori_signals_eval,ori_labels_eval, np.linspace(0, category_num-1,category_num,dtype=np.int64))

    signals_train = np.concatenate((signals_train, received_signals_train))
    labels_train = np.concatenate((labels_train, received_labels_train))
    signals_eval = np.concatenate((signals_eval, received_signals_eval))
    labels_eval = np.concatenate((labels_eval, received_labels_eval))

    label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels_train)
    opt = options.get_auto_options(opt, label_cnt_per, label_num, signals_train)
    train_sequences = np.linspace(0, len(labels_train)-1,len(labels_train),dtype=np.int64)
    eval_sequences = np.linspace(0, len(labels_eval)-1,len(labels_eval),dtype=np.int64)

    for epoch in range(opt.epochs):
        t1 = time.time()

        core.train(signals_train,labels_train,train_sequences)
        core.eval(signals_eval,labels_eval,eval_sequences)

        t2=time.time()
        if epoch+1==1:
            util.writelog('>>> per epoch cost time:'+str(round((t2-t1),2))+'s',opt,True)
    plot.draw_heatmap(core.confusion_mats[-1],opt,name = 'final')
    core.save_traced_net()

# -----------------------------server-----------------------------
from flask import Flask, request
import base64
import shutil

app = Flask(__name__)

key = '123456'
@app.route("/handlepost", methods=["POST"])
def handlepost():
    if request.form['token'] != key:
        return {'return':'token error'}

    if request.form['mode'] == 'clean':
        if os.path.isdir(opt.rec_tmp):
            shutil.rmtree(opt.rec_tmp)
        return {'return':'done'}

    if request.form['mode'] == 'send':
        data = request.form['data']
        util.makedirs(os.path.join(opt.rec_tmp, request.form['label']))
        util.savetxt(data, os.path.join(opt.rec_tmp, request.form['label'],util.randomstr(8)))
        return {'return':'done'}
    
    if request.form['mode'] == 'train':
        train(opt)
        file = util.loadfile(os.path.join(opt.save_dir,'model.pt'))
        file = base64.b64encode(file).decode('utf-8')
        heatmap = util.loadfile(os.path.join(opt.save_dir,'final_heatmap.png'))
        heatmap = base64.b64encode(heatmap).decode('utf-8')
        return {'return' : 'done',
                'report' : 'macro-prec,reca,F1,err,kappa:'+str(statistics.report(core.confusion_mats[-1])),
                'heatmap': heatmap,
                'network': file
                }

    return {'return':'error'}

app.run("0.0.0.0", port= 4000, debug=False)

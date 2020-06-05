import requests
import base64
import json
import os
import sys
sys.path.append("..")
from util import util


# ---------------------Parameter Init---------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--url',type=str,default="http://localhost:4000/handlepost", help='')
parser.add_argument('--token',type=str,default="123456", help='')
opt = parser.parse_args()

# -------Empty files that already exist on the server-------
data = {'token':opt.token,'mode': 'clean'}
r = requests.post(opt.url, data)
print(r.json())

# -----------load each samples and send them to server-----------
send_data_dir = './client_data/send_data'
labels = os.listdir(send_data_dir)
labels.sort()
for i in range(len(labels)):
    samples = os.listdir(os.path.join(send_data_dir,labels[i]))
    for j in range(len(samples)):
        print('send:',os.path.join(send_data_dir,labels[i],samples[j]))
        txt_data = util.loadtxt(os.path.join(send_data_dir,labels[i],samples[j]))
        data = {'token': opt.token,
                'mode' : 'send',
                'label': str(i),
                'data' : txt_data
                }
        r = requests.post(opt.url, data)
print(r.json())


"""Train and get network weight
return: {'return' : 'done',
        'report'  : 'macro-prec,reca,F1,err,kappa:'+str(statistics.report(core.confusion_mats[-1])),
        'heatmap' : heatmap, # encode by base64
        'network' : file     # encode by base64
        }
"""
data = {'token':opt.token,'mode': 'train'}
r = requests.post(opt.url, data)
rec_data = r.json()
print(rec_data['report'])

# save model.pt
util.makedirs('./client_data')
file = base64.b64decode(rec_data['network'])
util.savefile(file,'./client_data/model.pt')
# save heatmap.png
file = base64.b64decode(rec_data['heatmap'])
util.savefile(file,'./client_data/heatmap.png')

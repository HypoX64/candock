import os
import numpy as np

import sys
sys.path.append("..")
from util import util,options
from data import dataloader,statistics

from util import array_operation as arr

opt = options.Options()
opt.parser.add_argument('--dels',type=str,default='', help='which domains you want to del, eg. [0,1,2,3,4]')
opt.parser.add_argument('--keeps',type=str,default='', help='which domains you want to keep, and del other, eg. [0,1,2,3,4]')
opt.parser.add_argument('--foldbydomain',action='store_true', help='if specified, generate new fold index by domain, else 5-fold.')
opt = opt.getparse()
opt.dels = options.str2list(opt.dels,out_type='int')
opt.keeps = options.str2list(opt.keeps,out_type='int')

signals = np.load(os.path.join(opt.dataset_dir,'signals.npy'))
print(signals.shape)
labels = np.load(os.path.join(opt.dataset_dir,'labels.npy'))
print(labels.shape)
domains = np.load(os.path.join(opt.dataset_dir,'domains.npy'))
datas = [signals,labels,domains]

# del domain
del_indexs = []
if opt.keeps == []:
    for i in range(len(domains)):
        if domains[i] in opt.dels:
            del_indexs.append(i)
if opt.dels == []:
    for i in range(len(domains)):
        if domains[i] not in opt.keeps:
            del_indexs.append(i)
del_indexs = np.array(del_indexs)
signals = np.delete(signals,del_indexs, axis = 0)
labels  = np.delete(labels,del_indexs, axis = 0)
domains = np.delete(domains,del_indexs, axis = 0)
# for i in range(3): datas[i] = np.delete(datas[i],del_indexs, axis = 0)

# generate new fold index
# for i in range(3): datas[i] = datas[i][domains.argsort()]
signals = signals[domains.argsort()]
labels  = labels[domains.argsort()]
domains = domains[domains.argsort()]
domain_cnt,domain_num = statistics.domain_statistics(domains)
print('statistics:',domain_cnt,domain_num)
fold_indexs = []
fold_index = 0
if opt.foldbydomain:
    for i in range(domain_num-1):
        fold_index += domain_cnt[i]
        fold_indexs.append(fold_index)
    print('new index:',fold_indexs)
else:
    fold_domains = np.linspace(domain_num//5, domain_num-domain_num//5,4,dtype=np.int64)
    for i in range(domain_num):
        fold_index += domain_cnt[i]
        if i in fold_domains.tolist():
            fold_indexs.append(fold_index)
    print('new index:',fold_indexs)

print(signals.shape)
np.save(os.path.join(opt.save_dir,'signals.npy'), signals)
np.save(os.path.join(opt.save_dir,'labels.npy'), labels)
np.save(os.path.join(opt.save_dir,'domains.npy'), domains)
np.save(os.path.join(opt.save_dir,'index.npy'), np.array(fold_indexs))



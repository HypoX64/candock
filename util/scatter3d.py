import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

def label_statistics(labels):
    labels = (np.array(labels)).astype(np.int64)
    label_num = np.max(labels)+1
    label_cnt = np.zeros(label_num,dtype=np.int64)
    for i in range(len(labels)):
        label_cnt[labels[i]] += 1
    label_cnt_per = label_cnt/len(labels)
    return label_cnt,label_cnt_per,label_num

colors= ['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
markers = ['o','^','.',',','v','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']

def draw(data,opt):
    label_cnt,_,label_num = label_statistics(data[:,3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cnt = 0
    for i in range(label_num):
        ax.scatter(data[cnt:cnt+label_cnt[i],0], data[cnt:cnt+label_cnt[i],1], data[cnt:cnt+label_cnt[i],2],
            c = colors[i%10],marker = markers[i//10])
        cnt += label_cnt[i]

    plt.savefig(os.path.join(opt.save_dir,'scatter3d.png'))
    np.save(os.path.join(opt.save_dir,'scatter3d.npy'), data)
    plt.close('all')

def show(data):
    label_cnt,_,label_num = label_statistics(data[:,3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cnt = 0
    for i in range(label_num):
        ax.scatter(data[cnt:cnt+label_cnt[i],0], data[cnt:cnt+label_cnt[i],1], data[cnt:cnt+label_cnt[i],2],
            c = colors[i%10],marker = markers[i//10])
        cnt += label_cnt[i]

    plt.show()

if __name__ == '__main__':

    data = np.load('../checkpoints/au/scatter3d.npy')
    show(data)
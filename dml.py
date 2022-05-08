from posix import listdir
from scipy import signal
import torch
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
from util import util,dsp
from util import array_operation as arrop

class MyDataset(Dataset):

    def __init__(self,dataset_dir):
        self.filepaths,self.labels = [],[]
        filelabels = os.listdir(dataset_dir)
        filelabels = sorted(filelabels)
        for i in range(len(filelabels)):
            sub_files = os.listdir(os.path.join(dataset_dir,filelabels[i]))
            sub_files = sorted(sub_files)
            for sub_file in sub_files:
                self.filepaths.append(os.path.join(dataset_dir,filelabels[i],sub_file))
                self.labels.append(np.array(i))
        self.len = len(self.filepaths)

    def __getitem__(self, index):
        signal = np.load(self.filepaths[index])
        label = self.labels[index]
        signal = arrop.normliaze(signal,mode='5_95')
        signal = dsp.signal2spectrum(signal,128,108,n_downsample=2,log=False)
        label = torch.from_numpy(label)
        signal = torch.from_numpy(signal)
        return signal,label

    def __len__(self):
        return self.len

 
TrainDataset = MyDataset('./datasets/EarID/03_network/base/00/train')

loader = DataLoader(dataset=TrainDataset,
                    batch_size=32,
                    shuffle=True)

for i, data in enumerate(loader):
    signal,label = data
    print(label)
    print(signal)
    exit()
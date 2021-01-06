import os
import sys
import numpy as np
import torchvision
import torch.utils.data as data
from PIL import Image

sys.path.append("..")
from util import util
from util import array_operation as arr

save_dir = '../datasets/mnist2m_2d'
util.makedirs(save_dir)
# mnist
source_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,                                  # download it if you don't have it
)

mnist_signals = source_data.data.numpy()
mnist_signals = mnist_signals.astype(np.float32)
mnist_signals = ((mnist_signals)/256.0-0.1307)/0.3081
mnist_signals = mnist_signals.reshape(-1,1,28,28)
# mnist_signals = np.concatenate((mnist_signals,mnist_signals,mnist_signals),axis=1)
mnist_labels = source_data.targets.numpy()
mnist_labels = mnist_labels.reshape(-1,1)
print('mnist:',np.max(mnist_signals),np.min(mnist_signals))
print('mnist:',mnist_signals.shape)

target_image_root = './mnist_m'
data_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
f = open(data_list, 'r')
data_list = f.readlines()
f.close()

mnist_m_labels = []
mnist_m_signals = []
for data in data_list:
    mnist_m_labels.append(data[-2])
    mnist_m_signals.append(np.array(Image.open(os.path.join(target_image_root,'mnist_m_train', data[:-3])).convert('L')))

mnist_m_signals = (np.array(mnist_m_signals)).astype(np.float32).reshape(-1,1,32,32)

mnist_m_signals = ((mnist_m_signals[:,:,2:30,2:30]-128)/128)
mnist_m_labels = np.array(mnist_m_labels).reshape(-1,1)
print('mnist_m:',np.max(mnist_m_signals),np.min(mnist_m_signals))
print('mnist_m:',mnist_m_signals.shape)

indexs = np.array([len(mnist_labels)])
domains = np.concatenate((np.zeros(len(mnist_labels),dtype=np.int64),np.ones(len(mnist_m_labels),dtype=np.int64)),axis=0)
signals = np.concatenate((mnist_signals,mnist_m_signals),axis=0)
labels = np.concatenate((mnist_labels,mnist_m_labels),axis=0)

np.save(os.path.join(save_dir,'index'), indexs)
np.save(os.path.join(save_dir,'domains'), domains)
np.save(os.path.join(save_dir,'signals'), signals)
np.save(os.path.join(save_dir,'labels'), labels)
# candock
[这原本是一个用于记录毕业设计的日志仓库](<https://github.com/HypoX64/candock/tree/Graduation_Project>)，其目的是尝试多种不同的深度神经网络结构(如LSTM,ResNet,DFCNN等)对单通道EEG进行自动化睡眠阶段分期.<br>目前，项目重点将转变为如何建立一个通用的一维时序信号分析,分类框架.<br>它将包含多种网络结构，并提供数据预处理,读取,训练,评估,测试等功能.<br>
一些训练时的输出样例: [heatmap](./image/heatmap_eg.png)  [running_err](./image/running_err_eg.png)  [log.txt](./docs/log_eg.txt)

## 注意
为了适应新的项目，代码已被大幅更改，不能确保仍然能正常运行如sleep-edfx等睡眠数据集，如果仍然需要运行，请参照下文按照输入格式标准自行加载数据，如果有时间我会修复这个问题。
当然，如果需要加载睡眠数据集也可以直接使用[老的版本](https://github.com/HypoX64/candock/tree/f24cc44933f494d2235b3bf965a04cde5e6a1ae9)

## Getting Started
### Prerequisites
- Linux, Windows,mac
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- Pytroch 1.0+
### Dependencies
This code depends on torchvision, numpy, scipy , matplotlib,available via pip install.<br>
For example:<br>

```bash
pip3 install matplotlib
```
### Clone this repo:
```bash
git clone https://github.com/HypoX64/candock
cd candock
```
### Download dataset and pretrained-model
[[Google Drive]](https://drive.google.com/open?id=1NTtLmT02jqlc81lhtzQ7GlPK8epuHfU5)   [[百度云,y4ks]](https://pan.baidu.com/s/1WKWZL91SekrSlhOoEC1bQA)

* This datasets consists of signals.npy(shape:18207, 1, 2000) and labels.npy(shape:18207) which can be loaded by "np.load()".
* samples:18207,  channel:1,  length of each sample:2000,  class:50
* Top1 err: 2.09%
### Train
```bash
python3 train.py --label 50 --input_nc 1 --dataset_dir ./datasets/simple_test --save_dir ./checkpoints/simple_test --model_name micro_multi_scale_resnet_1d --gpu_id 0 --batchsize 64 --k_fold 5
```
* For more [options](./options.py).
#### Use your own data to train
* step1: Generate signals.npy and labels.npy in the following format.
```python
#1.type:numpydata   signals:np.float64   labels:np.int64
#2.shape  signals:[num,ch,length]    labels:[num]
#num:samples_num, ch :channel_num,  num:length of each sample
#for example:
signals = np.zeros((10,1,10),dtype='np.float64')
labels = np.array([0,0,0,0,0,1,1,1,1,1])      #0->class0    1->class1
```
* step2: input  ```--dataset_dir your_dataset_dir``` when running code.
### Test
```bash
python3 simple_test.py --label 50 --input_nc 1 --model_name micro_multi_scale_resnet_1d --gpu_id 0
```
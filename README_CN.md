<div align="center">    
<img src="./imgs/compare.png " alt="image" style="zoom:100%;" />
</div>

# candock

| [English](./README.md) | 中文版 |<br>
一个通用的一维时序信号分析,分类框架.<br>
它将包含多种网络结构，并提供数据预处理,读取,训练,评估,测试等功能.<br>
一些训练时的输出样例: [heatmap](./image/heatmap_eg.png)  [running_err](./image/running_err_eg.png)  [log.txt](./docs/log_eg.txt)<br>
目前支持的网络结构:<br>
>1d
>
>>lstm, cnn_1d, resnet18_1d, resnet34_1d, multi_scale_resnet_1d, micro_multi_scale_resnet_1d


>2d(stft spectrum)
>
>>mobilenet, resnet18, resnet50, resnet101, densenet121, densenet201, squeezenet, dfcnn, multi_scale_resnet,

## 关于EEG睡眠分期数据的实例
为了适应新的项目，代码已被大幅更改，不能正常运行如sleep-edfx等睡眠数据集，如果仍然需要运行，请参照下文按照输入格式标准自行加载数据，如果有时间我会修复这个问题。
当然，如果需要加载睡眠数据集也可以直接使用[老的版本](https://github.com/HypoX64/candock/tree/f24cc44933f494d2235b3bf965a04cde5e6a1ae9)<br>
感谢[@swalltail99](https://github.com/swalltail99)指出的错误，为了适应sleep-edfx数据集的读取，使用这个版本的代码时，请安装mne==0.18.0<br>

```bash
pip install mne==0.18.0
```

## 入门
### 前提要求
- Linux, Windows,mac
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3
- Pytroch 1.0+
### 依赖
This code depends on torchvision, numpy, scipy , matplotlib, available via pip install.<br>
For example:<br>

```bash
pip3 install matplotlib
```
### 克隆仓库:
```bash
git clone https://github.com/HypoX64/candock
cd candock
```
### 下载数据集以及预训练模型
[[Google Drive]](https://drive.google.com/open?id=1NTtLmT02jqlc81lhtzQ7GlPK8epuHfU5)   [[百度云,y4ks]](https://pan.baidu.com/s/1WKWZL91SekrSlhOoEC1bQA)

* 数据集包括 signals.npy(shape:18207, 1, 2000) 以及 labels.npy(shape:18207) 可以使用"np.load()"加载
* 样本量:18207,  通道数:1,  每个样本的长度:2000,  总类别数:50
* Top1 err: 2.09%
### 训练
```bash
python3 train.py --label 50 --input_nc 1 --dataset_dir ./datasets/simple_test --save_dir ./checkpoints/simple_test --model_name micro_multi_scale_resnet_1d --gpu_id 0 --batchsize 64 --k_fold 5
# 如果需要使用cpu进行训练, 请输入 --gpu_id -1
```
* 更多可选参数 [options](./util/options.py).
### 测试
```bash
python3 simple_test.py --label 50 --input_nc 1 --model_name micro_multi_scale_resnet_1d --gpu_id 0
# 如果需要使用cpu进行训练, 请输入 --gpu_id -1
```

## 使用自己的数据进行训练
* step1: 按照如下格式生成 signals.npy 以及 labels.npy.
```python
#1.type:numpydata   signals:np.float64   labels:np.int64
#2.shape  signals:[num,ch,length]    labels:[num]
#num:samples_num, ch :channel_num,  num:length of each sample
#for example:
signals = np.zeros((10,1,10),dtype='np.float64')
labels = np.array([0,0,0,0,0,1,1,1,1,1])      #0->class0    1->class1
```
* step2: 输入  ```--dataset_dir "your_dataset_dir"``` 当运行代码的时候.


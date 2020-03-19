# candock
[这原本是一个用于记录毕业设计的日志仓库](<https://github.com/HypoX64/candock/tree/Graduation_Project>)，其目的是尝试多种不同的深度神经网络结构(如LSTM,ResNet,DFCNN等)对单通道EEG进行自动化睡眠阶段分期.<br>目前，毕业设计已经完成，我将继续跟进这个项目。项目重点将转变为如何将代码进行实际应用，我们将考虑运算量与准确率之间的平衡。另外，将提供一些预训练的模型便于使用。<br>同时我们相信这些代码也可以用于其他生理信号(如ECG,EMG等)的分类.希望这将有助于您的研究或项目.<br>
![image](https://github.com/HypoX64/candock/blob/master/image/compare.png)

## 注意
为了适应新的项目，代码已被大幅更改，不能确保仍然能正常运行如sleep-edfx等睡眠数据集，如果仍然需要运行，请按照输入格式标准自行加载数据，如果有时间我会修复这个问题。
当然，也可以直接使用[老的版本](https://github.com/HypoX64/candock/tree/f24cc44933f494d2235b3bf965a04cde5e6a1ae9)
```python
'''
#数据输入格式
change your own data to train
but the data needs meet the following conditions: 
1.type   numpydata  signals:np.float16  labels:np.int16
2.shape             signals:[num,ch,length]   labels:[num]
'''
```

## 如何运行
如果你需要运行这些代码（训练自己的模型或者使用预训练模型对自己的数据进行预测）请进入以下页面<br>
[How to run codes](https://github.com/HypoX64/candock/blob/master/how_to_run.md)<br>

## 数据集
使用了两个公开的睡眠数据集进行训练，分别是:   [[CinC Challenge 2018]](https://physionet.org/physiobank/database/challenge/2018/#files)     [[sleep-edfx]](https://www.physionet.org/physiobank/database/sleep-edfx/) <br>
对于CinC Challenge 2018数据集,我们仅使用其C4-M1通道, 对于sleep-edfx与sleep-edf数据集,使用Fpz-Cz通道<br>
注意：<br>
1.如果需要获得其他EEG通道的预训练模型，这需要下载这两个数据集并使用train.py完成训练。当然，你也可以使用自己的数据训练模型。<br>
2.对于sleep-edfx数据集,我们仅仅截取了入睡前30分钟到醒来后30分钟之间的睡眠区间作为读入数据(实验结果中用select sleep time 进行标注),目的是平衡各睡眠时期的比例并加快训练速度.<br>

## 一些说明
* 数据预处理<br>

  1.降采样：CinC Challenge 2018数据集的EEG信号将被降采样到100HZ<br>

  2.归一化处理：我们推荐每个受试者的EEG信号均采用5th-95th分位数归一化，即第5%大的数据为0，第95%大的数据为1。注意：所有预训练模型均按照这个方法进行归一化后训练得到<br>

  3.将读取的数据分割为30s/Epoch作为一个输入，每个输入包含3000个数据点。睡眠阶段标签为5个分别是N3,N2,N1,REM,W.每个Epoch的数据将对应一个标签。标签映射：N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4<br>

  4.数据集扩充：训练时，对每一个Epoch的数据均要进行随机切，随机翻转，随机改变信号幅度等操作<br>

  5.对于不同的网络结构,对原始eeg信号采取了预处理,使其拥有不同的shape:<br>
  LSTM:将30s的eeg信号进行FIR带通滤波,获得θ,σ,α,δ,β波,并将它们进行连接后作为输入数据<br>
  CNN_1d类(标有1d的网络):没有什么特别的操作，其实就是把图像领域的各种模型换成Conv1d之后拿过来用而已<br>
  DFCNN类(就是科大讯飞的那种想法，先转化为频谱图，然后直接用图像分类的各种模型):将30s的eeg信号进行短时傅里叶变换,并生成频谱图作为输入,并使用图像分类网络进行分类。我们不推荐使用这种方法，因为转化为频谱图需要耗费较大的运算资源。<br>

* EEG频谱图<br>
  这里展示5个睡眠阶段对应的频谱图,它们依次是Wake, Stage 1, Stage 2, Stage 3, REM<br>
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_Wake.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_Stage1.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_Stage2.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_Stage3.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_REM.png)<br>

* multi_scale_resnet_1d 网络结构<br>
  该网络参考[geekfeiw / Multi-Scale-1D-ResNet](https://github.com/geekfeiw/Multi-Scale-1D-ResNet)     这个网络将被我们命名为micro_multi_scale_resnet_1d<br>
  修改后的[网络结构](https://github.com/HypoX64/candock/blob/master/image/multi_scale_resnet_1d_network.png)<br>

* 关于交叉验证<br>为了更好的进行实际应用，我们将使用受试者交叉验证。即训练集和验证集的数据来自于不同的受试者。值得注意的是sleep-edfx数据集中每个受试者均有两个样本，我们视两个样本为同一个受试者，很多paper忽略了这一点，手动滑稽。<br>

* 关于评估指标<br>
  对于各睡眠阶段标签:  Accuracy = (TP+TN)/(TP+FN+TN+FP)   Recall = sensitivity = (TP)/(TP+FN)<br>
  对于总体:   Top1 err.    Kappa    另外对Acc与Re做平均<br>
  特别说明：这项分类任务中样本标签分布及不平衡，为了更具说服力，我们的平均并不加权。

## 部分实验结果
该部分将持续更新... ...<br>
[[Confusion matrix]](https://github.com/HypoX64/candock/blob/master/confusion_mat)<br>

#### Subject Cross-Validation Results
特别说明：这项分类任务中样本标签分布及不平衡，我们对分类损失函数中的类别权重进行了魔改，这将使得Average Recall得到小幅提升，但同时整体error也将提升.若使用默认权重，Top1 err.至少下降5%,但这会导致数据占比极小的N1时期的recall猛跌20%，这绝对不是我们在实际应用中所希望看到的。下面给出的结果均是使用魔改后的权重得到的。<br>
* [sleep-edfx](https://www.physionet.org/physiobank/database/sleep-edfx/)  ->sample size = 197, select sleep time

| Network                     | Parameters | Top1.err. | Avg. Acc. | Avg. Re. | Need to extract feature |
| --------------------------- | ---------- | --------- | --------- | -------- | ----------------------- |
| lstm                        | 1.25M      | 26.32%    | 89.47%    | 68.57%   | Yes                     |
| micro_multi_scale_resnet_1d | 2.11M      | 25.33%    | 89.87%    | 72.61%   | No                      |
| resnet18_1d                 | 3.85M      | 24.21%    | 90.31%    | 72.87%   | No                      |
| multi_scale_resnet_1d       | 8.42M      | 24.01%    | 90.40%    | 72.37%   | No                      |
* [CinC Challenge 2018](https://physionet.org/physiobank/database/challenge/2018/#files)  ->sample size = 994

| Network                     | Parameters | Top1.err. | Avg. Acc. | Avg. Re. | Need to extract feature |
| --------------------------- | ---------- | --------- | --------- | -------- | ----------------------- |
| lstm                        | 1.25M      | 26.85%    | 89.26%    | 71.39%   | Yes                     |
| micro_multi_scale_resnet_1d | 2.11M      | 27.01%    | 89.20%    | 73.12%   | No                      |
| resnet18_1d                 | 3.85M      | 25.84%    | 89.66%    | 73.32%   | No                      |
| multi_scale_resnet_1d       | 8.42M      | 25.27%    | 89.89%    | 73.63%   | No                      |
```

```
# candock
这是一个用于记录毕业设计的日志仓库，其目的是尝试多种不同的深度神经网络结构(如LSTM,RESNET,DFCNN等)对单通道EEG进行自动化睡眠阶段分期.我们相信这些代码同时可以用于其他生理信号(如ECG,EMG等)的分类.希望这将有助于您的研究.<br>
## 数据集
使用了三个睡眠数据集进行测试,分别是: [CinC Challenge 2018](https://physionet.org/physiobank/database/challenge/2018/#files)   [sleep-edf](https://www.physionet.org/physiobank/database/sleep-edf/)   [sleep-edfx](https://www.physionet.org/physiobank/database/sleep-edfx/) <br>
对于CinC Challenge 2018数据集,使用其C4-M1通道<br>对于sleep-edfx与sleep-edf数据集,使用Fpz-Cz通道<br>
值得注意的是:sleep-edfx是sleep-edf的扩展版本.<br>

## 一些说明
* 数据集分割<br>
  读取数据集各样本后分割为30s/Epoch作为一个输入,共有5个标签,分别是Sleep stage 3,2,1,R,W,将分割后的eeg信号与睡眠阶段标签进行对应后,打乱其顺序,并将80%的数据用于训练,20%的数据用于测试.
* 数据预处理<br>
  对于不同的网络结构,对原始eeg信号采取了预处理,使其拥有不同的shape:<br>
  LSTM:将30s的eeg信号进行FIR带通滤波,获得θ,σ,α,δ,β波,并将它们进行连接后作为输入数据<br>
  resnet_1d:这里使用resnet的一维形式进行实验,(修改nn.Conv2d为nn.Conv1d).<br>
  DFCNN:将30s的eeg信号进行短时傅里叶变换,并生成频谱图作为输入,并使用resnet网络进行分类.<br>
* EEG频谱图<br>
  这里展示5个睡眠阶段对应的频谱图,它们依次是wake, stage 1, stage 2, stage 3, REM
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_wake.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_N1.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_N2.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_N3.png)
  ![image](https://github.com/HypoX64/candock/blob/master/image/spectrum_REM.png)
* 关于代码<br>
  目前的代码仍然在不断修改与更新中,不能确保其能工作.详细内容将会在毕业设计完成后抽空更新.<br>
## 部分实验结果
该部分将持续更新... ...
* sleep-edf<br>
  [[Confusion matrix]](https://github.com/HypoX64/candock/blob/master/image/confusion_mat)<br>
  
  | Network        | Label average recall | Label average accuracy | error rate |
  | :------------- | :------------------- | ---------------------- | ---------- |
  | lstm           | 0.8342               | 0.9611                 | 0.0974     |
  | resnet18_1d    | 0.8434               | 0.9627                 | 0.093      |
  | DFCNN+resnet18 | 0.8567               | 0.9663                 | 0.0842     |

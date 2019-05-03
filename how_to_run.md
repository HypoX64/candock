## Prerequisites
- Linux, Windows,mac
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3.5+
- Pytroch 1.0+

## Dependencies
This code depends on torchvision, numpy, scipy, h5py, matplotlib, mne , requests, hashlib, available via pip install.<br>
For example:<br>

```bash
pip3 install matplotlib
```
But for mne, you have to run:<br>
```bash
pip3 install -U https://api.github.com/repos/mne-tools/mne-python/zipball/master
```

## Getting Started
### Clone this repo:
```bash
git clone https://github.com/HypoX64/candock
cd candock
```
### Train
* download datasets
```bash
python3 download_dataset.py
```
* choose your options and run
```bash
python3 train.py --dataset_dir './datasets/sleep-edfx/' --dataset_name sleep-edfx --signal_name 'EEG Fpz-Cz' --sample_num 10 --model_name lstm --batchsize 64 --network_save_freq 5 --epochs 20 --lr 0.0005 --BID 5_95_th --select_sleep_time --cross_validation subject
```
* Notes<br>
If want to use cpu to train or test, please use --no_cuda

### Simple Test
* download pretrained model & simple test data  [Google Drive](https://drive.google.com/open?id=1pup2_tZFGQQwB-hoXRjpMxiD4Vmpn0Lf)
* choose your options and run
```bash
python3 simple_test.py
```
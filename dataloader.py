import os
import time
import random

import scipy.io as sio
import numpy as np
import h5py
import mne

import dsp
import transformer


def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def reducesample(data,mult):
    return data[::mult]

# delete uesless label
def del_UND(signals,stages):
    stages_copy = stages.copy()
    cnt = 0
    for i in range(len(stages_copy)):
        if stages_copy[i] == 5 :
            signals = np.delete(signals,i-cnt,axis =0)
            stages = np.delete(stages,i-cnt,axis =0)
            cnt += 1
    return signals,stages

def connectdata(signal,stage,signals=[],stages=[]):
    if signals == []:
        signals =signal.copy()
        stages =stage.copy()
    else:
        signals=np.concatenate((signals, signal), axis=0)
        stages=np.concatenate((stages, stage), axis=0)
    return signals,stages

#load one subject data form cc2018
def loaddata_cc2018(filedir,filename,signal_name,BID,filter = True):
    dirpath = os.path.join(filedir,filename)
    #load signal
    hea_path = os.path.join(dirpath,os.path.basename(dirpath)+'.hea')
    signal_path = os.path.join(dirpath,os.path.basename(dirpath)+'.mat')
    signal_names = []
    for i,line in enumerate(open(hea_path),0):
        if i!=0:
            line=line.strip()
            signal_names.append(line.split()[8])
    mat = sio.loadmat(signal_path)
    signals = mat['val'][signal_names.index(signal_name)]
    if filter:
        signals = dsp.BPF(signals,200,0.2,50,mod = 'fir')
    #load stage
    stagepath = os.path.join(dirpath,os.path.basename(dirpath)+'-arousal.mat')
    mat=h5py.File(stagepath,'r')
    # N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4  UND->5
    N3 = mat['data']['sleep_stages']['nonrem3'][0]
    N2 = mat['data']['sleep_stages']['nonrem2'][0]
    N1 = mat['data']['sleep_stages']['nonrem1'][0]
    REM = mat['data']['sleep_stages']['rem'][0]
    W = mat['data']['sleep_stages']['wake'][0]
    UND = mat['data']['sleep_stages']['undefined'][0]
    stages = N3*0 + N2*1 + N1*2 + REM*3 + W*4 + UND*5
    #resample
    signals = reducesample(signals,2)
    stages = reducesample(stages,2)
    #trim
    signals = trimdata(signals,3000)
    stages = trimdata(stages,3000)
    #30s per lable
    signals = signals.reshape(-1,3000)
    stages = stages[::3000]
    #Balance individualized differences
    signals = transformer.Balance_individualized_differences(signals, BID)
    #del UND
    signals,stages = del_UND(signals, stages)

    return signals.astype(np.float16),stages.astype(np.int16)

#load one subject data form sleep-edfx
def loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time):
    filenum = filename[2:6]
    filenames = os.listdir(filedir)
    for filename in filenames:
        if str(filenum) in filename and 'Hypnogram' in filename:
            f_stage_name = filename
        if str(filenum) in filename and 'PSG' in filename:
            f_signal_name = filename

    raw_data= mne.io.read_raw_edf(os.path.join(filedir,f_signal_name),preload=True)
    raw_annot = mne.read_annotations(os.path.join(filedir,f_stage_name))
    eeg = raw_data.pick_channels([signal_name]).to_data_frame().values.T
    eeg = eeg.reshape(-1)

    raw_data.set_annotations(raw_annot, emit_warning=False)
    #N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4  other->UND->5
    event_id = {'Sleep stage 4': 0,
                  'Sleep stage 3': 0,
                  'Sleep stage 2': 1,
                  'Sleep stage 1': 2,
                  'Sleep stage R': 3,
                  'Sleep stage W': 4,
                  'Sleep stage ?': 5,
                  'Movement time': 5}
    events, _ = mne.events_from_annotations(
        raw_data, event_id=event_id, chunk_duration=30.)

    stages = []
    signals =[]
    for i in range(len(events)-1):
        stages.append(events[i][2])
        signals.append(eeg[events[i][0]:events[i][0]+3000])
    stages=np.array(stages)
    signals=np.array(signals)

    # #select sleep time 
    if select_sleep_time:
        if 'SC' in f_signal_name:
            signals = signals[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]
            stages = stages[np.clip(int(raw_annot[0]['duration'])//30-60,0,9999999):int(raw_annot[-2]['onset'])//30+60]

    signals,stages = del_UND(signals, stages)
    print('shape:',signals.shape,stages.shape)

    signals = transformer.Balance_individualized_differences(signals, BID)

    return signals.astype(np.float16),stages.astype(np.int16)

#load all data in datasets
def loaddataset(filedir,dataset_name,signal_name,num,BID,select_sleep_time,shuffle = True):
    print('load dataset, please wait...')
    filenames = os.listdir(filedir)
    if shuffle:
        random.shuffle(filenames)
    signals=[]
    stages=[]

    if dataset_name == 'cc2018':
        if num > len(filenames):
            num = len(filenames)
            print('num of dataset is:',num)

        for cnt,filename in enumerate(filenames[:num],0):
            try:
                signal,stage = loaddata_cc2018(filedir,filename,signal_name,BID = BID)
                signals,stages = connectdata(signal,stage,signals,stages)
            except Exception as e:
                print(filename,e)

    elif dataset_name in ['sleep-edfx','sleep-edf']:
        if num > 197:
            num = 197
        if dataset_name == 'sleep-edf':
            filenames = ['SC4002E0-PSG.edf','SC4012E0-PSG.edf','SC4102E0-PSG.edf','SC4112E0-PSG.edf',
            'ST7022J0-PSG.edf','ST7052J0-PSG.edf','ST7121J0-PSG.edf','ST7132J0-PSG.edf']        
        cnt = 0
        for filename in filenames:
            if 'PSG' in filename:
                signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time)
                signals,stages = connectdata(signal,stage,signals,stages)
                cnt += 1
                if cnt == num:
                    break
    return signals,stages
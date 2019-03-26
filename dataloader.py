import scipy.io as sio
import numpy as np
import h5py
import os
import time
import torch
import random
import DSP
# import pyedflib
import mne

# CinC_Challenge_2018
def loadstages(dirpath):
    filepath = os.path.join(dirpath,os.path.basename(dirpath)+'-arousal.mat')
    mat=h5py.File(filepath,'r')
    # N3(S4+S3)->0  N2->1  N1->2  REM->3  W->4  UND->5
    #print(mat.keys())
    N3 = mat['data']['sleep_stages']['nonrem3'][0]
    N2 = mat['data']['sleep_stages']['nonrem2'][0]
    N1 = mat['data']['sleep_stages']['nonrem1'][0]
    REM = mat['data']['sleep_stages']['rem'][0]
    W = mat['data']['sleep_stages']['wake'][0]
    UND = mat['data']['sleep_stages']['undefined'][0]
    stages = N3*0 + N2*1 + N1*2 + REM*3 + W*4 + UND*5
    return stages

def loadsignals(dirpath,name):
    hea_path = os.path.join(dirpath,os.path.basename(dirpath)+'.hea')
    signal_path = os.path.join(dirpath,os.path.basename(dirpath)+'.mat')
    signal_names = []
    for i,line in enumerate(open(hea_path),0):
        if i!=0:
            line=line.strip()
            signal_names.append(line.split()[8])
    mat = sio.loadmat(signal_path)
    return mat['val'][signal_names.index(name)]

def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def reducesample(data,mult):
    return data[::mult]

def loaddata(dirpath,signal_name,BID = 'median',filter = True):
    #load
    signals = loadsignals(dirpath,signal_name)
    if filter:
        signals = DSP.BPF(signals,200,0.2,50,mod = 'fir')
    stages = loadstages(dirpath)
    #resample
    signals = reducesample(signals,2)
    stages = reducesample(stages,2)
    #Balance individualized differences
    if BID == 'median':
        signals = (signals*8/(np.median(abs(signals)))).astype(np.int16)
    elif BID == 'std':
        signals = (signals*55/(np.std(signals))).astype(np.int16)
    #trim
    signals = trimdata(signals,3000)
    stages = trimdata(stages,3000)
    #30s per lable
    signals = signals.reshape(-1,3000)
    stages = stages[::3000]
    #del UND
    stages_copy = stages.copy()
    cnt = 0
    for i in range(len(stages_copy)):
        if stages_copy[i] == 5 :
            signals = np.delete(signals,i-cnt,axis =0)
            stages = np.delete(stages,i-cnt,axis =0)
            cnt += 1
    # print(stages.shape,signals.shape)
    return signals,stages


def stage_str2int(stagestr):
    if stagestr == 'Sleep stage 3' or stagestr == 'Sleep stage 4':
        stage = 0
    elif stagestr == 'Sleep stage 2':
        stage = 1
    elif stagestr == 'Sleep stage 1':
        stage = 2
    elif stagestr == 'Sleep stage R':
        stage = 3
    elif stagestr == 'Sleep stage W':
        stage = 4
    elif stagestr == 'Sleep stage ?'or stagestr == 'Movement time':
        stage = 5
    return stage

def loaddata_sleep_edf(filedir,filenum,signal_name,BID = 'median',filter = True):
    filenames = os.listdir(filedir)
    for filename in filenames:
        if str(filenum) in filename and 'Hypnogram' in filename:
            f_stage_name = filename
        if str(filenum) in filename and 'PSG' in filename:
            f_signal_name = filename
    print(f_stage_name)

    raw_data= mne.io.read_raw_edf(os.path.join(filedir,f_signal_name),preload=True)
    raw_annot = mne.read_annotations(os.path.join(filedir,f_stage_name))
    eeg = raw_data.pick_channels(['EEG Fpz-Cz']).to_data_frame().values.T.reshape(-1)

    raw_data.set_annotations(raw_annot, emit_warning=False)
    event_id = {'Sleep stage 4': 0,
                  'Sleep stage 3': 0,
                  'Sleep stage 2': 1,
                  'Sleep stage 1': 2,
                  'Sleep stage R': 3,
                  'Sleep stage W': 4,
                  'Sleep stage ?': 5,
                  'Sleep stage Movement time': 5}
    events, _ = mne.events_from_annotations(
        raw_data, event_id=event_id, chunk_duration=30.)
    events = np.array(events)

    signals=trimdata(eeg,3000)
    signals = signals.reshape(-1,3000)
    stages = events[:,2]
    print(signals.shape,events.shape)
    # stages = stages[0:signals.shape[0]]

    stages_copy = stages.copy()
    cnt = 0
    for i in range(len(stages_copy)):
        if stages_copy[i] == 5 :
            signals = np.delete(signals,i-cnt,axis =0)
            stages = np.delete(stages,i-cnt,axis =0)
            cnt += 1




    # print(f_signal_name)
    '''
    f_stage = pyedflib.EdfReader(os.path.join(filedir,f_stage_name))
    annotations = f_stage.readAnnotations()
    number_of_annotations = f_stage.annotations_in_file
    end_duration = int(annotations[0][number_of_annotations-1])+int(annotations[1][number_of_annotations-1])
    stages = np.zeros(end_duration//30, dtype=int)
    # print(number_of_annotations)
    for i in range(number_of_annotations):
        stages[int(annotations[0][i])//30:(int(annotations[0][i])+int(annotations[1][i]))//30] = stage_str2int(annotations[2][i])

    f_signal = pyedflib.EdfReader(os.path.join(filedir,f_signal_name))
    signals = f_signal.readSignal(0)
    signals=trimdata(signals,3000)
    signals = signals.reshape(-1,3000)
    stages = stages[0:signals.shape[0]]

    # #select sleep time 
    # signals = signals[np.clip(int(annotations[1][0])//30-60,0,9999999):int(annotations[0][number_of_annotations-2])//30+60]
    # stages = stages[np.clip(int(annotations[1][0])//30-60,0,9999999):int(annotations[0][number_of_annotations-2])//30+60]
    
    #del UND
    stages_copy = stages.copy()
    cnt = 0
    for i in range(len(stages_copy)):
        if stages_copy[i] == 5 :
            signals = np.delete(signals,i-cnt,axis =0)
            stages = np.delete(stages,i-cnt,axis =0)
            cnt += 1
    '''
    return signals.astype(np.int16),stages.astype(np.int16)


def loaddataset(filedir,dataset_name = 'CinC_Challenge_2018',signal_name = 'C4-M1',num = 100 ,BID = 'median',shuffle = True):
    print('load dataset, please wait...')
    filenames = os.listdir(filedir)

    if shuffle:
        random.shuffle(filenames)

    if dataset_name == 'CinC_Challenge_2018':
        if num > len(filenames):
            num = len(filenames)
            print('num of dataset is:',num)
        for i,filename in enumerate(filenames[:num],0):

            try:
                signal,stage = loaddata(os.path.join(filedir,filename),signal_name,BID)
                if i == 0:
                    signals =signal.copy()
                    stages =stage.copy()
                else:
                    signals=np.concatenate((signals, signal), axis=0)
                    stages=np.concatenate((stages, stage), axis=0)
            except Exception as e:
                print(filename,e)
    elif dataset_name == 'sleep-edfx':
        cnt = 0
        for filename in filenames:
            if 'PSG' in filename:
                signal,stage = loaddata_sleep_edf(filedir,filename[2:6],signal_name = 'FPZ-CZ')
                if cnt == 0:
                    signals =signal.copy()
                    stages =stage.copy()
                else:
                    signals=np.concatenate((signals, signal), axis=0)
                    stages=np.concatenate((stages, stage), axis=0)
                cnt += 1
                if cnt == num:
                    break
    return signals,stages
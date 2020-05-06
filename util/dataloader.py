import os
import time
import random

import scipy.io as sio
import numpy as np

from . import dsp,transformer,statistics
# import dsp
# import transformer
# import statistics


def trimdata(data,num):
    return data[:num*int(len(data)/num)]

def reducesample(data,mult):
    return data[::mult]

def balance_label(signals,labels):

    label_sta,_,label_num = statistics.label_statistics(labels)
    ori_length = len(labels)
    max_label_length = max(label_sta)
    signals = signals[labels.argsort()]
    labels = labels[labels.argsort()]

    if signals.ndim == 2:
        new_signals = np.zeros((max_label_length*label_num,signals.shape[1]), dtype=signals.dtype)
    elif signals.ndim == 3:
        new_signals = np.zeros((max_label_length*label_num,signals.shape[1],signals.shape[2]), dtype=signals.dtype)
    new_labels = np.zeros((max_label_length*label_num), dtype=labels.dtype)
    new_signals[:ori_length] = signals
    new_labels[:ori_length] = labels
    del(signals)
    del(labels)

    cnt = ori_length
    for label in range(len(label_sta)):
        if label_sta[label] < max_label_length:
            if label == 0:
                start = 0
            else:
                start = np.sum(label_sta[:label])
            end = np.sum(label_sta[:label+1])-1

            for i in range(max_label_length-label_sta[label]):
                new_signals[cnt] = new_signals[random.randint(start,end)]
                new_labels[cnt] = label
                cnt +=1
    return new_signals,new_labels


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
    #30s per label
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

    stages = [];signals =[]
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
def loaddataset(opt,shuffle = False): 
    filedir=opt.dataset_dir
    dataset_name = opt.dataset_name
    signal_name = opt.signal_name
    num = opt.sample_num
    BID = opt.BID
    select_sleep_time = opt.select_sleep_time

    print('load dataset, please wait...')

    signals_train=[];labels_train=[];signals_test=[];labels_test=[] 

    if dataset_name == 'cc2018':
        import h5py
        filenames = os.listdir(filedir)
        if not opt.no_shuffle:
            random.shuffle(filenames)
        else:
            filenames.sort()

        if num > len(filenames):
            num = len(filenames)
            print('num of dataset is:',num)

        for cnt,filename in enumerate(filenames[:num],0):
            signal,stage = loaddata_cc2018(filedir,filename,signal_name,BID = BID)
            if cnt < round(num*0.8) :
                signals_train,labels_train = connectdata(signal,stage,signals_train,labels_train)
            else:
                signals_test,labels_test = connectdata(signal,stage,signals_test,labels_test)
        print('train subjects:',round(num*0.8),'test subjects:',round(num*0.2))

    elif dataset_name == 'sleep-edfx':
        import mne
        if num > 197:
            num = 197

        filenames_sc_train = ['SC4001E0-PSG.edf', 'SC4002E0-PSG.edf', 'SC4011E0-PSG.edf', 'SC4012E0-PSG.edf', 'SC4021E0-PSG.edf', 'SC4022E0-PSG.edf', 'SC4031E0-PSG.edf', 'SC4032E0-PSG.edf', 'SC4041E0-PSG.edf', 'SC4042E0-PSG.edf', 'SC4051E0-PSG.edf', 'SC4052E0-PSG.edf', 'SC4061E0-PSG.edf', 'SC4062E0-PSG.edf', 'SC4071E0-PSG.edf', 'SC4072E0-PSG.edf', 'SC4081E0-PSG.edf', 'SC4082E0-PSG.edf', 'SC4091E0-PSG.edf', 'SC4092E0-PSG.edf', 'SC4101E0-PSG.edf', 'SC4102E0-PSG.edf', 'SC4111E0-PSG.edf', 'SC4112E0-PSG.edf', 'SC4121E0-PSG.edf', 'SC4122E0-PSG.edf', 'SC4131E0-PSG.edf', 'SC4141E0-PSG.edf', 'SC4142E0-PSG.edf', 'SC4151E0-PSG.edf', 'SC4152E0-PSG.edf', 'SC4161E0-PSG.edf', 'SC4162E0-PSG.edf', 'SC4171E0-PSG.edf', 'SC4172E0-PSG.edf', 'SC4181E0-PSG.edf', 'SC4182E0-PSG.edf', 'SC4191E0-PSG.edf', 'SC4192E0-PSG.edf', 'SC4201E0-PSG.edf', 'SC4202E0-PSG.edf', 'SC4211E0-PSG.edf', 'SC4212E0-PSG.edf', 'SC4221E0-PSG.edf', 'SC4222E0-PSG.edf', 'SC4231E0-PSG.edf', 'SC4232E0-PSG.edf', 'SC4241E0-PSG.edf', 'SC4242E0-PSG.edf', 'SC4251E0-PSG.edf', 'SC4252E0-PSG.edf', 'SC4261F0-PSG.edf', 'SC4262F0-PSG.edf', 'SC4271F0-PSG.edf', 'SC4272F0-PSG.edf', 'SC4281G0-PSG.edf', 'SC4282G0-PSG.edf', 'SC4291G0-PSG.edf', 'SC4292G0-PSG.edf', 'SC4301E0-PSG.edf', 'SC4302E0-PSG.edf', 'SC4311E0-PSG.edf', 'SC4312E0-PSG.edf', 'SC4321E0-PSG.edf', 'SC4322E0-PSG.edf', 'SC4331F0-PSG.edf', 'SC4332F0-PSG.edf', 'SC4341F0-PSG.edf', 'SC4342F0-PSG.edf', 'SC4351F0-PSG.edf', 'SC4352F0-PSG.edf', 'SC4362F0-PSG.edf', 'SC4371F0-PSG.edf', 'SC4372F0-PSG.edf', 'SC4381F0-PSG.edf', 'SC4382F0-PSG.edf', 'SC4401E0-PSG.edf', 'SC4402E0-PSG.edf', 'SC4411E0-PSG.edf', 'SC4412E0-PSG.edf', 'SC4421E0-PSG.edf', 'SC4422E0-PSG.edf', 'SC4431E0-PSG.edf', 'SC4432E0-PSG.edf', 'SC4441E0-PSG.edf', 'SC4442E0-PSG.edf', 'SC4451F0-PSG.edf', 'SC4452F0-PSG.edf', 'SC4461F0-PSG.edf', 'SC4462F0-PSG.edf', 'SC4471F0-PSG.edf', 'SC4472F0-PSG.edf', 'SC4481F0-PSG.edf', 'SC4482F0-PSG.edf', 'SC4491G0-PSG.edf', 'SC4492G0-PSG.edf', 'SC4501E0-PSG.edf', 'SC4502E0-PSG.edf', 'SC4511E0-PSG.edf', 'SC4512E0-PSG.edf', 'SC4522E0-PSG.edf', 'SC4531E0-PSG.edf', 'SC4532E0-PSG.edf', 'SC4541F0-PSG.edf', 'SC4542F0-PSG.edf', 'SC4551F0-PSG.edf', 'SC4552F0-PSG.edf', 'SC4561F0-PSG.edf', 'SC4562F0-PSG.edf', 'SC4571F0-PSG.edf', 'SC4572F0-PSG.edf', 'SC4581G0-PSG.edf', 'SC4582G0-PSG.edf', 'SC4591G0-PSG.edf', 'SC4592G0-PSG.edf', 'SC4601E0-PSG.edf', 'SC4602E0-PSG.edf', 'SC4611E0-PSG.edf', 'SC4612E0-PSG.edf', 'SC4621E0-PSG.edf', 'SC4622E0-PSG.edf', 'SC4631E0-PSG.edf', 'SC4632E0-PSG.edf']
        filenames_sc_test = ['SC4641E0-PSG.edf', 'SC4642E0-PSG.edf', 'SC4651E0-PSG.edf', 'SC4652E0-PSG.edf', 'SC4661E0-PSG.edf', 'SC4662E0-PSG.edf', 'SC4671G0-PSG.edf', 'SC4672G0-PSG.edf', 'SC4701E0-PSG.edf', 'SC4702E0-PSG.edf', 'SC4711E0-PSG.edf', 'SC4712E0-PSG.edf', 'SC4721E0-PSG.edf', 'SC4722E0-PSG.edf', 'SC4731E0-PSG.edf', 'SC4732E0-PSG.edf', 'SC4741E0-PSG.edf', 'SC4742E0-PSG.edf', 'SC4751E0-PSG.edf', 'SC4752E0-PSG.edf', 'SC4761E0-PSG.edf', 'SC4762E0-PSG.edf', 'SC4771G0-PSG.edf', 'SC4772G0-PSG.edf', 'SC4801G0-PSG.edf', 'SC4802G0-PSG.edf', 'SC4811G0-PSG.edf', 'SC4812G0-PSG.edf', 'SC4821G0-PSG.edf', 'SC4822G0-PSG.edf']
        filenames_st_train = ['ST7011J0-PSG.edf', 'ST7012J0-PSG.edf', 'ST7021J0-PSG.edf', 'ST7022J0-PSG.edf', 'ST7041J0-PSG.edf', 'ST7042J0-PSG.edf', 'ST7051J0-PSG.edf', 'ST7052J0-PSG.edf', 'ST7061J0-PSG.edf', 'ST7062J0-PSG.edf', 'ST7071J0-PSG.edf', 'ST7072J0-PSG.edf', 'ST7081J0-PSG.edf', 'ST7082J0-PSG.edf', 'ST7091J0-PSG.edf', 'ST7092J0-PSG.edf', 'ST7101J0-PSG.edf', 'ST7102J0-PSG.edf', 'ST7111J0-PSG.edf', 'ST7112J0-PSG.edf', 'ST7121J0-PSG.edf', 'ST7122J0-PSG.edf', 'ST7131J0-PSG.edf', 'ST7132J0-PSG.edf', 'ST7141J0-PSG.edf', 'ST7142J0-PSG.edf', 'ST7151J0-PSG.edf', 'ST7152J0-PSG.edf', 'ST7161J0-PSG.edf', 'ST7162J0-PSG.edf', 'ST7171J0-PSG.edf', 'ST7172J0-PSG.edf', 'ST7181J0-PSG.edf', 'ST7182J0-PSG.edf', 'ST7191J0-PSG.edf', 'ST7192J0-PSG.edf']
        filenames_st_test = ['ST7201J0-PSG.edf', 'ST7202J0-PSG.edf', 'ST7211J0-PSG.edf', 'ST7212J0-PSG.edf', 'ST7221J0-PSG.edf', 'ST7222J0-PSG.edf', 'ST7241J0-PSG.edf', 'ST7242J0-PSG.edf']

        for filename in filenames_sc_train[:round(num*153/197*0.8)]:
            signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time)
            signals_train,labels_train = connectdata(signal,stage,signals_train,labels_train)

        for filename in filenames_st_train[:round(num*44/197*0.8)]:
            signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time)
            signals_train,labels_train = connectdata(signal,stage,signals_train,labels_train)
        
        for filename in filenames_sc_test[:round(num*153/197*0.2)]:
            signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time)
            signals_test,labels_test = connectdata(signal,stage,signals_test,labels_test)

        for filename in filenames_st_test[:round(num*44/197*0.2)]:
            signal,stage = loaddata_sleep_edfx(filedir,filename,signal_name,BID,select_sleep_time)
            signals_test,labels_test = connectdata(signal,stage,signals_test,labels_test)

        print('---------Each subject has two sample---------',
            '\nTrain samples_SC/ST:',round(num*153/197*0.8),round(num*44/197*0.8),
            '\nTest samples_SC/ST:',round(num*153/197*0.2),round(num*44/197*0.2))
    
    elif dataset_name == 'preload':
        if opt.separated:
            signals_train = np.load(filedir+'/signals_train.npy')
            labels_train = np.load(filedir+'/labels_train.npy')
            signals_test = np.load(filedir+'/signals_test.npy')
            labels_test = np.load(filedir+'/labels_test.npy')
        else:
            signals = np.load(filedir+'/signals.npy') 
            labels = np.load(filedir+'/labels.npy')
            if not opt.no_shuffle:
                transformer.shuffledata(signals,labels)

    if opt.separated:
        return signals_train,labels_train,signals_test,labels_test
    else:
        return signals,labels
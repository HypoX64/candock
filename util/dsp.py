import scipy.signal
import scipy.fftpack as fftpack
import numpy as np

b1 = scipy.signal.firwin(31, [0.5, 4], pass_zero=False,fs=100)
b2 = scipy.signal.firwin(31, [4,8], pass_zero=False,fs=100)
b3 = scipy.signal.firwin(31, [8,12], pass_zero=False,fs=100)
b4 = scipy.signal.firwin(31, [12,16], pass_zero=False,fs=100)
b5 = scipy.signal.firwin(31, [16,45], pass_zero=False,fs=100)

def getfir_b(fc1,fc2,fs):
    if fc1==0.5 and fc2==4 and fs==100:
        b=b1
    elif fc1==4 and fc2==8 and fs==100:
        b=b2
    elif fc1==8 and fc2==12 and fs==100:
        b=b3
    elif fc1==12 and fc2==16 and fs==100:
        b=b4
    elif fc1==16 and fc2==45 and fs==100:
        b=b5
    else:
        b=scipy.signal.firwin(51, [fc1, fc2], pass_zero=False,fs=fs)
    return b


def BPF(signal,fs,fc1,fc2,mod = 'fir'):
    if mod == 'fft':
        length=len(signal)#get N
        k1=int(fc1*length/fs)#get k1=Nw1/fs
        k2=int(fc2*length/fs)#get k1=Nw1/fs
        #FFT
        signal_fft=fftpack.fft(signal)
        #Frequency truncation
        signal_fft[0:k1]=0+0j
        signal_fft[k2:length-k2]=0+0j
        signal_fft[length-k1:length]=0+0j
        #IFFT
        signal_ifft=fftpack.ifft(signal_fft)
        result = signal_ifft.real
    else:
        b=getfir_b(fc1,fc2,fs)
        result = scipy.signal.lfilter(b, 1, signal)
    return result

def getfeature(signal,mod = 'fft',ch_num = 5):
    result=[]
    signal =signal - np.mean(signal)
    eeg=signal

    beta=BPF(eeg,100,16,45,mod)    # β
    theta=BPF(eeg,100,4,8,mod)   #θ
    sigma=BPF(eeg,100,12,16,mod) #σ spindle
    alpha=BPF(eeg,100,8,12,mod)  #α
    delta=BPF(eeg,100,0.5,4,mod) #δ
   
    result.append(beta) 
    result.append(theta)  
    result.append(sigma)
    result.append(alpha)
    result.append(delta)

    if ch_num == 6:
        fft = abs(fftpack.fft(eeg))
        fft = fft - np.median(fft)
        result.append(fft)

    result=np.array(result)
    result=result.reshape(ch_num*len(signal),)
    return result

# def signal2spectrum(data):
#     # window : ('tukey',0.5) hann

#     zxx = scipy.signal.stft(data, fs=100, window='hann', nperseg=1024, noverlap=1024-12, nfft=1024, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)[2]
#     zxx =np.abs(zxx)[:512]
#     spectrum=np.zeros((256,251))
#     spectrum[0:128]=zxx[0:128]
#     spectrum[128:192]=zxx[128:256][::2]
#     spectrum[192:256]=zxx[256:512][::4]
#     spectrum = np.log(spectrum+1)
#     return spectrum

def signal2spectrum(data):
    # window : ('tukey',0.5) hann

    zxx = scipy.signal.stft(data, fs=100, window='hann', nperseg=1024, noverlap=1024-24, nfft=1024, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)[2]
    zxx =np.abs(zxx)[:512]
    spectrum=np.zeros((256,126))
    spectrum[0:128]=zxx[0:128]
    spectrum[128:192]=zxx[128:256][::2]
    spectrum[192:256]=zxx[256:512][::4]
    spectrum = np.log(spectrum+1)
    return spectrum
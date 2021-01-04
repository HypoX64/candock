import scipy.signal
import scipy.fftpack as fftpack
import numpy as np
import pywt
import cv2
from . import array_operation as arr

def sin(f,fs,time,theta=0):
    x = np.linspace(0, 2*np.pi*f*time, int(fs*time))
    return np.sin(x+theta)

def downsample(signal,fs1=0,fs2=0,alpha=0,mod = 'just_down'):
    if alpha == 0:
        alpha = int(fs1/fs2)
    if mod == 'just_down':
        return signal[::alpha]
    elif mod == 'avg':
        result = np.zeros(int(len(signal)/alpha))
        for i in range(int(len(signal)/alpha)):
            result[i] = np.mean(signal[i*alpha:(i+1)*alpha])
        return result       

def medfilt(signal,x):
    return scipy.signal.medfilt(signal,x)

def cleanoffset(signal):
    return signal - np.mean(signal)

def showfreq(signal,fs,fc=0,db=False):
    """
    return f,fft
    """
    if fc==0:
        kc = int(len(signal)/2)
    else:   
        kc = int(len(signal)/fs*fc)
    signal_fft = np.abs(scipy.fftpack.fft(signal))
    f = np.linspace(0,fs/2,num=int(len(signal_fft)/2))
    out_f = f[:kc]
    out_fft = signal_fft[0:int(len(signal_fft)/2)][:kc]
    if db:
        out_fft = 20*np.log10(out_fft/np.max(out_fft))
        out_fft = out_fft-np.max(out_fft)
        np.clip(out_fft,-100,0)
    return out_f,out_fft

def fft(signal,half = True,db=True,normliaze=True):
    signal_fft = np.abs(scipy.fftpack.fft(signal))
    if half:
        signal_fft = signal_fft[:len(signal_fft)//2]
    if db:
        signal_fft = 20*np.log10(signal_fft)
    if normliaze:
        signal_fft = arr.normliaze(signal_fft,mode = '5_95',truncated = 4)
    return signal_fft

def bpf(signal, fs, fc1, fc2, numtaps=3, mode='iir'):
    if mode == 'iir':
        b,a = scipy.signal.iirfilter(numtaps, [fc1,fc2], fs=fs)
    elif mode == 'fir':
        b = scipy.signal.firwin(numtaps, [fc1, fc2], pass_zero=False,fs=fs)
        a = 1       
    return scipy.signal.lfilter(b, a, signal)

def wave_filter(signal,wave,level,usedcoeffs):
    '''
    wave       : wavelet name string, wavelet(eg. dbN symN haar gaus mexh)
    level      : decomposition level 
    usedcoeffs : coeff used for reconstruction  eg. when level = 6 usedcoeffs=[1,1,0,0,0,0,0] : reconstruct signal with cA6, cD6
    '''
    coeffs = pywt.wavedec(signal, wave, level=level)
    #[cAn, cDn, cDn-1, …, cD2, cD1]
    for i in range(len(usedcoeffs)):
        if usedcoeffs[i] == 0:
            coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wave, mode='symmetric', axis=-1)

def fft_filter(signal,fs,fc=[],type = 'bandpass'):
    '''
    signal: Signal
    fs: Sampling frequency
    fc: [fc1,fc2...] Cut-off frequency 
    type: bandpass | bandstop
    '''
    k = []
    N=len(signal)#get N

    for i in range(len(fc)):
        k.append(int(fc[i]*N/fs))

    #FFT
    signal_fft=scipy.fftpack.fft(signal)
    #Frequency truncation

    if type == 'bandpass':
        a = np.zeros(N)
        for i in range(int(len(fc)/2)):
            a[k[2*i]:k[2*i+1]] = 1
            a[N-k[2*i+1]:N-k[2*i]] = 1
    elif type == 'bandstop':
        a = np.ones(N)
        for i in range(int(len(fc)/2)):
            a[k[2*i]:k[2*i+1]] = 0
            a[N-k[2*i+1]:N-k[2*i]] = 0
    signal_fft = a*signal_fft
    signal_ifft=scipy.fftpack.ifft(signal_fft)
    result = signal_ifft.real
    return result

def rms(signal):
    signal = signal.astype('float64')
    return np.mean((signal*signal))**0.5

def energy(signal,kernel_size,stride,padding = 0):
    _signal = np.zeros(len(signal)+padding)
    _signal[0:len(signal)] = signal
    signal = _signal
    out_len = int((len(signal)+1-kernel_size)/stride)
    energy = np.zeros(out_len)
    for i in range(out_len):
        energy[i] = rms(signal[i*stride:i*stride+kernel_size]) 
    return energy

def signal2spectrum(data,stft_window_size,stft_stride,cwt_wavename,cwt_scale_num,n_downsample=1, log = True, log_alpha = 0.1, mod = 'stft'):
    # window : ('tukey',0.5) hann
    if n_downsample != 1:
        data = downsample(data,alpha=n_downsample)

    if mod == 'stft':
        zxx = scipy.signal.stft(data, window='hann', nperseg=stft_window_size,noverlap=stft_window_size-stft_stride)[2]
        spectrum = np.abs(zxx)

        if log:
            spectrum = np.log1p(spectrum)
            h = spectrum.shape[0]
            x = np.linspace(h*log_alpha, h-1,num=h+1,dtype=np.int64)
            index = (np.log1p(x)-np.log1p(h*log_alpha))/(np.log1p(h)-np.log1p(h*log_alpha))*h

            spectrum_new = np.zeros_like(spectrum)
            for i in range(h):
                spectrum_new[int(index[i]):int(index[i+1])] = spectrum[i]
            spectrum = spectrum_new

    if mod == 'cwt':

        fc = pywt.central_frequency(cwt_wavename)
        cparam = 2 * fc * cwt_scale_num
        scales = cparam / np.arange(cwt_scale_num, 1, -1)  
        cwtmatr, frequencies = pywt.cwt(data, scales, cwt_wavename,method='fft')
        spectrum = np.abs(cwtmatr)
        spectrum = cv2.resize(spectrum,(cwt_scale_num-1,cwt_scale_num-1),interpolation=cv2.INTER_AREA)

    return spectrum
import scipy.signal
import scipy.fftpack as fftpack
import numpy as np

def sin(f,fs,time):
    x = np.linspace(0, 2*np.pi*f*time, fs*time)
    return np.sin(x)

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

def bpf_fir(signal,fs,fc1,fc2,numtaps=101):
    b=scipy.signal.firwin(numtaps, [fc1, fc2], pass_zero=False,fs=fs)
    result = scipy.signal.lfilter(b, 1, signal)
    return result

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

def signal2spectrum(data,window_size, stride, n_downsample=1, log = True, log_alpha = 0.1):
    # window : ('tukey',0.5) hann
    if n_downsample != 1:
        data = downsample(data,alpha=n_downsample)

    zxx = scipy.signal.stft(data, window='hann', nperseg=window_size,noverlap=window_size-stride)[2]
    spectrum = np.abs(zxx)

    if log:
        spectrum = np.log1p(spectrum)
        h = window_size//2+1
        x = np.linspace(h*log_alpha, h-1,num=h+1,dtype=np.int64)
        index = (np.log1p(x)-np.log1p(h*log_alpha))/(np.log1p(h)-np.log1p(h*log_alpha))*h

        spectrum_new = np.zeros_like(spectrum)
        for i in range(h):
            spectrum_new[int(index[i]):int(index[i+1])] = spectrum[i]
        spectrum = spectrum_new
        spectrum = (spectrum-0.05)/0.25

    else:
        spectrum = (spectrum-0.02)/0.05

    return spectrum
# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
from __future__ import division
import os.path as op
import numpy
from . import sigproc
from scipy.fftpack import dct
from typing import BinaryIO, Optional, Tuple, Union
import matplotlib.pyplot as plt
def maptoheat3(x, cmap_style):
    x = numpy.maximum(x, 0)
    x /= numpy.max(x)+ 1e-5

    #plt.figure()
    plt.matshow(x, aspect='auto', cmap=cmap_style, origin='lower')
    #plt.show()

def hist_normalization(img, a=0, b=1.0):
    # get max and min
    c = img.min()
    d = img.max()

    out = img.copy()
    if(d-c) > 0:
        coef = (b-a) / (d - c)

        # normalization
        out = coef * (out - c) + a
        out[out < a] = a
        out[out > b] = b
    else:
        print("WTF", c, d)

    return out

def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.

    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

def mfcc_n(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    numcep=13,
    nfilt=26,
    nfft=None,
    lowfreq=0,
    highfreq=None,
    preemph=0.970000029,
    ceplifter=22,
    appendEnergy=True,
    winfunc=lambda x:numpy.ones((x,))
):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat

def mfcc_m(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    numcep=13,
    nfilt=26,
    nfft=None,
    lowfreq=0,
    highfreq=None,
    preemph=0.970000029,
    ceplifter=22,
    appendEnergy=True,
    winfunc=lambda x:numpy.ones((x,))
):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is None, which uses the calculate_nfft function to choose the smallest size that does not drop sample data.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    nfft = nfft or calculate_nfft(samplerate, winlen)
    feat, energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
    feat = lifter(feat,ceplifter)
    #if appendEnergy: feat[:,0] = numpy.log(energy) # replace first cepstral coefficient with log of frame energy
    return feat, numpy.log(energy).reshape([-1,1])

def stft_analysis(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.970000029,
    winfunc=lambda x: numpy.ones((x,))
):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    frames = sigproc.framesig(signal, preemph, winlen * samplerate, winstep * samplerate, winfunc)
    energy = numpy.sum(numpy.square(frames),1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log


    complex_spec = numpy.fft.rfft(frames, nfft)

    return numpy.absolute(complex_spec), numpy.angle(complex_spec)


def logfftmeg(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.970000029,
    winfunc=lambda x: numpy.ones((x,))
):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    frames = sigproc.framesig(signal, preemph, winlen * samplerate, winstep * samplerate, winfunc)
    energy = numpy.sum(numpy.square(frames),1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    pspec = sigproc.powspec(frames, nfft)
    #feat = pspec[:,:nfilt]
    feat = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # if feat is zero, we get problems with log

    return numpy.log(feat)

def fbank(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.970000029,
    winfunc=lambda x: numpy.ones((x,))
):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq= highfreq or samplerate/2
    frames = sigproc.framesig(signal, preemph, winlen * samplerate, winstep * samplerate, winfunc)
    energy = numpy.sum(numpy.square(frames),1) # this stores the total energy in each frame
    energy = numpy.where(energy == 0,numpy.finfo(float).eps,energy) # if energy is zero, we get problems with log

    pspec = sigproc.powspec(frames, nfft)
    #maptoheat3(pspec,'jet')
    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)

    feat = numpy.dot(pspec,fb) # compute the filterbank energies
    feat = numpy.where(feat == 0,numpy.finfo(float).eps,feat) # if feat is zero, we get problems with log
    #maptoheat3(feat, 'jet')
    '''
    fb_i = numpy.reciprocal(fb)
    fb_inv = numpy.linalg.pinv(fb)
    e = numpy.dot(fb,fb_inv)
    ei = numpy.dot(fb_inv,fb)
    fb_t = numpy.transpose(fb)
    pspec_t = numpy.transpose(pspec)
    feat_t = numpy.dot(fb_t, pspec_t)
    fb_t_i = numpy.linalg.pinv(fb_t)
    e_t = numpy.dot(fb_t_i, fb_t)
    '''
    return feat,energy

def logfbank(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.97,
    winfunc=lambda x: numpy.ones((x,))
):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
    return numpy.log(feat)

def ssc(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.97,
    winfunc=lambda x: numpy.ones((x,))
):
    """Compute Spectral Subband Centroid features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    highfreq = highfreq or samplerate/2
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = sigproc.powspec(frames, nfft)
    pspec = numpy.where(pspec == 0,numpy.finfo(float).eps,pspec) # if things are all zeros we get problems

    fb = get_filterbanks(nfilt,nfft,samplerate,lowfreq,highfreq)
    feat = numpy.dot(pspec,fb.T) # compute the filterbank energies
    r = numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(pspec,1)),(numpy.size(pspec,0),1))

    return numpy.dot(pspec * r, fb.T) / feat

def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(
    nfilt=20,
    nfft=512,
    samplerate=16000,
    lowfreq=0,
    highfreq=None
):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size (nfft/2 + 1) * nfilt containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = nfft*mel2hz(melpoints)/samplerate

    bankSize = nfft//2+1
    fbank = numpy.zeros([bankSize, nfilt])
    for i in range(0, bankSize):
        for j in range(0, nfilt):
            highlope = (i - bin[j]) / (bin[j+1]-bin[j])
            loslope = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
            fbank[i, j] = max(min(highlope, loslope), 0.0)
    return fbank

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, winLen):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if winLen < 1:
        raise ValueError('winLen must be an integer >= 1')
    numFrames = len(feat)
    numFeats = feat.shape[1]
    denominator = (winLen * (winLen + 1) * (2 * winLen + 1)) / 3.0
    delta_feat = numpy.zeros(feat.shape)
    for t in range(numFrames):
        for f in range(numFeats):
            for w in range(1, winLen+1):
                delta_feat[t, f] += w * (feat[t + min((numFrames - t - 1), w), f] -
                                            feat[t - min(t, w), f])
            delta_feat[t, f] /= denominator
    return delta_feat

def get_waveform(
    path_or_fp: Union[str, BinaryIO], dtype=numpy.int16, normalization=True
) -> Tuple[numpy.ndarray, int]:
    if isinstance(path_or_fp, str):
        ext = op.splitext(op.basename(path_or_fp))[1]
        if ext not in {".flac", ".wav"}:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile to load WAV/FLAC file")

    waveform, sample_rate = sf.read(path_or_fp, dtype=dtype)
    if not normalization:
        waveform *= 2 ** 15  # denormalized to 16-bit signed integers
    return waveform, sample_rate

def wav_to_spectrogram(wavfile, twindow, tshift):
    ''' compute speech spectrogram

    @wavfile : input wave file
    @twindow : window lengh (ms)
    @tshift : window shift (ms)

    return: spectral data, wav data points, wav sample rate
    '''
    wavdata, wav_rate = get_waveform(wavfile)
    wav_len = len(wavdata)
    window = twindow * wav_rate // 1000
    shift = tshift * wav_rate // 1000
    dim = window // 2 + 1
    n_frames = (wav_len - (window - shift) + shift - 1) // shift

    x = numpy.linspace(0, window - 1, window, dtype=numpy.float)
    w = 0.54 - 0.46 * numpy.cos(numpy.pi * 2 * x / (window - 1))

    fbank = numpy.zeros((n_frames, dim), dtype=numpy.complex)

    start = 0
    for i in range(n_frames - 1):
        fbank[i] = numpy.fft.fft(wavdata[start:start + window] * w)[0:dim]
        start += shift
    # the last frame
    padding = window + (n_frames - 1) * shift - wav_len
    last_data = numpy.pad(wavdata[start:wav_len], (0, padding), 'constant', constant_values=0)
    fbank[n_frames - 1] = numpy.fft.fft(last_data * w)[0:dim]

    return fbank, wav_len, wav_rate


def spectrogram_to_wav(fbank, angs, data_len, rate, win, shift):
    ''' restore wav data from speech spectrogram

    @fbank : input speech power spectrogram
    @angs  : input referenced speech spectrogram phase angles
    @data_len : output wave data length
    @rate : output wave sample rate
    @win : window lengh in points
    @shift : window shift points

    return: wav data
    '''
    x = numpy.linspace(0, win - 1, win, dtype=numpy.float)
    w = 0.54 - 0.46 * numpy.cos(numpy.pi * 2 * x / (win - 1))

    winfreq = fbank * numpy.exp(1j * angs)
    winfreq = numpy.concatenate((winfreq, numpy.conjugate(numpy.flip(winfreq[:, 1:-1], 1))), axis=1)
    windata = numpy.real(numpy.fft.ifft(winfreq))

    n_frames = fbank.shape[0]
    align_len = win + (n_frames - 1) * shift
    outdata = numpy.zeros(align_len, dtype=numpy.float)
    scaledata = numpy.zeros(align_len, dtype=numpy.float)

    start = 0
    for i in range(n_frames - 1):
        outdata[start:start + win] += windata[i]
        scaledata[start:start + win] += w
        start += shift

    outdata = outdata / scaledata

    return numpy.round(outdata[0:data_len]).astype(numpy.int16)

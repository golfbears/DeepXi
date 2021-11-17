import functools
import os
import collections
import contextlib
import sys
import wave
import tensorflow as tf
import librosa
from tensorflow.python.ops.signal import window_ops
import numpy as np
import matplotlib.pylab as plt
import padasip as pa
import soundfile as sf
from mcra.mcra123 import imcra
import webrtcvad

class tensor_polar():
    def __init__(self, N_d, N_s, K, f_s):
        """
        Argument/s:
            N_d - window duration (samples).
            N_s - window shift (samples).
            K - number of frequency bins.
            f_s - sampling frequency.
        """
        self.N_d = N_d
        self.N_s = N_s
        self.K = K
        self.f_s = f_s
        self.W = functools.partial(window_ops.hamming_window,
                                   periodic=False)
        self.ten = tf.cast(10.0, tf.float32)
        self.one = tf.cast(1.0, tf.float32)

    def polar_analysis(self, x):
        """
        Polar-form acoustic-domain analysis.

        Argument/s:
            x - waveform.

        Returns:
            Short-time magnitude and phase spectrums.
        """
        STFT = tf.signal.stft(x, self.N_d, self.N_s, self.K,
                              window_fn=self.W, pad_end=True)
        return tf.abs(STFT), tf.math.angle(STFT)


    def polar_synthesis(self, STMS, STPS):
        """
        Polar-form acoustic-domain synthesis.

        Argument/s:
            STMS - short-time magnitude spectrum.
            STPS - short-time phase spectrum.

        Returns:
            Waveform.
        """
        STFT = tf.cast(STMS, tf.complex64) * tf.exp(1j * tf.cast(STPS, tf.complex64))
        return tf.signal.inverse_stft(STFT, self.N_d, self.N_s, self.K, tf.signal.inverse_stft_window_fn(self.N_s, self.W))

        # creation of data
        # these two function supplement your online measurment
def measure_x():
    # it produces input vector of size 3
    x = np.random.random(3)
    return x


def measure_d(x):
    # meausure system output
    d = 2 * x[0] + 1 * x[1] - 1.5 * x[2]
    return d

def measure_n(x, length):
    ln = len(x)
    if ln == 0:
        o = np.random.random(length)
    elif len(x) < length:
        b = np.random.random(length-len(x))
        o = np.concatenate((x,b))
    else:
        o = x[-length:]
    return o
def measure_int(x, length):
    ln = len(x)
    if ln == 0:
        o = np.random.random(length)*32767
    elif len(x) < length:
        b = np.random.random(length-len(x))*x[0]
        o = np.concatenate((x,b))
    else:
        o = x[-length:]
    return o.astype(np.int16)

def sample_lms_rt(x0, d_t, analysis_path, laps):
    # x0 = track[:, 1]
    # d_t = np.roll(track[:, 3],40)
    #x0 = mix1
    #d_t = noise
    # d_t = np.roll(mix,20)
    int_d_t = np.floor(d_t * 32767).astype(np.int16)

    # d_o = measure_n(x0)

    filt = pa.filters.FilterLMS(laps, mu=0.03)
    log_d = np.zeros(len(x0))
    for i, k in enumerate(x0):
        d_o = measure_n(d_t[:i + 1], laps)
        '''
        int_d_o = measure_int(int_d_t[:i + 1])
        spic = int_d_o.tobytes()
        is_speech = vad.is_speech(spic,sr)
        if not is_speech:
            filt.adapt(d_o, k)
        '''
        #filt.adapt(d_o, k)
        filt.adapt(k, d_o)
        y = filt.predict(d_o)
        log_d[i] = y

        ### show results
    for i in range(laps):
        log_d[i] = x0[i]
    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.title("Adaptation")
    plt.xlabel("samples - k")
    plt.plot(log_d, "b", label="predict - target")
    # plt.plot(x0, "g", label="x - input")
    plt.legend()
    plt.subplot(312);
    plt.title("Filter error")
    plt.xlabel("samples - k")
    # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
    plt.plot(x0, "g", label="noisy - input")
    # plt.plot(d_t, "b", label="d - input")
    plt.legend()
    plt.legend()
    plt.subplot(313)
    plt.title("Filter error")
    plt.xlabel("samples - k")
    # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
    plt.plot(x0 - log_d, "g", label="error between predict and input")
    # plt.plot(d_t, "b", label="d - input")
    plt.legend()
    plt.tight_layout()
    plt.show()
    path1 = os.path.join(analysis_path, 'predict' + '.wav')
    sf.write(path1, log_d, 16000)
    path1 = os.path.join(analysis_path, 'error' + '.wav')
    sf.write(path1, x0 - log_d, 16000)
    path1 = os.path.join(analysis_path, 'noisy' + '.wav')
    sf.write(path1, x0, 16000)
    '''    
    N = 100
    log_d = np.zeros(N)
    log_y = np.zeros(N)
    filt = pa.filters.FilterLMS(3, mu=1.)
    for k in range(N):
        # measure input
        x = measure_x()
        # predict new value
        y = filt.predict(x)
        # do the important stuff with prediction output
        pass
        # measure output
        d = measure_d(x)
        # update filter
        filt.adapt(d, x)
        # log values
        log_d[k] = d
        log_y[k] = y

    ### show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211);
    plt.title("Adaptation");
    plt.xlabel("samples - k")
    plt.plot(log_d, "b", label="d - target")
    plt.plot(log_y, "g", label="y - output");
    plt.legend()
    plt.subplot(212);
    plt.title("Filter error");
    plt.xlabel("samples - k")
    plt.plot(10 * np.log10((log_d - log_y) ** 2), "r", label="e - error [dB]")
    plt.legend();
    plt.tight_layout();
    plt.show()
    '''

def sample_nlms_rt(x0, d_t, analysis_path, laps):
    # x0 = track[:, 1]
    # d_t = np.roll(track[:, 3],40)
    #x0 = mix1
    #d_t = noise
    # d_t = np.roll(mix,20)

    D=200
    x = pa.input_from_history(x0,laps)[:-D]
    d = x0[laps+D-1]

    f = pa.filters.FilterNLMS(n=laps,mu=0.01,w='zeros')
    yp,e,w = f.run(d,x)
        ### show results
    #for i in range(laps):
    #    log_d[i] = x0[i]
    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.title("Adaptation")
    plt.xlabel("samples - k")
    plt.plot(yp, "b", label="predict - target")
    # plt.plot(x0, "g", label="x - input")
    plt.legend()
    plt.subplot(312);
    plt.title("Filter error")
    plt.xlabel("samples - k")
    # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
    plt.plot(e, "g", label="noisy - input")
    # plt.plot(d_t, "b", label="d - input")
    plt.legend()
    plt.legend()
    plt.subplot(313)
    plt.title("Filter error")
    plt.xlabel("samples - k")
    # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
    plt.plot(w, "g", label="error between predict and input")
    # plt.plot(d_t, "b", label="d - input")
    plt.legend()
    plt.tight_layout()
    plt.show()
    path1 = os.path.join(analysis_path, 'nlms_predict' + '.wav')
    sf.write(path1, yp, 16000)
    path1 = os.path.join(analysis_path, 'nlms_error' + '.wav')
    sf.write(path1, e, 16000)
    path1 = os.path.join(analysis_path, 'nlms_noisy' + '.wav')
    sf.write(path1, w, 16000)
    '''    
    N = 100
    log_d = np.zeros(N)
    log_y = np.zeros(N)
    filt = pa.filters.FilterLMS(3, mu=1.)
    for k in range(N):
        # measure input
        x = measure_x()
        # predict new value
        y = filt.predict(x)
        # do the important stuff with prediction output
        pass
        # measure output
        d = measure_d(x)
        # update filter
        filt.adapt(d, x)
        # log values
        log_d[k] = d
        log_y[k] = y

    ### show results
    plt.figure(figsize=(15, 9))
    plt.subplot(211);
    plt.title("Adaptation");
    plt.xlabel("samples - k")
    plt.plot(log_d, "b", label="d - target")
    plt.plot(log_y, "g", label="y - output");
    plt.legend()
    plt.subplot(212);
    plt.title("Filter error");
    plt.xlabel("samples - k")
    plt.plot(10 * np.log10((log_d - log_y) ** 2), "r", label="e - error [dB]")
    plt.legend();
    plt.tight_layout();
    plt.show()
    '''

if __name__ == '__main__':

    filt_bins = 30
    vad = webrtcvad.Vad(2)
    sample_rate = 16000
    frame_duration = 10  # ms
    frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
    print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))

    noise_dir = '/home/devpath/datasets/noise'
    int_track, int_sr = librosa.load(os.path.join(noise_dir, '48ksrate_3ksin_5min_wav.wav'),sr=16000, mono=False)
    #n_track, n_sr = sf.read(os.path.join(noise_dir, '48ksrate_3ksin_5min_wav.wav'), samplerate=16000)
    t_polar = tensor_polar(400, 160, 512, 16000)


    P_d = np.mean(np.square(int_track[0]), 0)  # avera

    # define path
    r_dir = '/home/devpath/wcs/noisetracking'
    home_dir = '/home/devpath/wcs/noisetracking/clean'
    analysis_path = os.path.join(r_dir,'enhance')
    f_list = os.listdir(home_dir)
    low_fr = np.ones(17, float) * 2
    mid_fr = np.ones(48, float) * 2.8
    hi_fr = np.ones(192, float) * 5
    delta = np.concatenate((low_fr, mid_fr, hi_fr))

    for file in f_list:
        if not file.__contains__('.wav'):
            continue
        track, sr = sf.read(os.path.join(home_dir,file))
        P_s = np.mean(np.square(track), 0)  # average power of clean speech.
        P_s1 = np.square(np.max(track))
        P_d1 = np.square(np.max(int_track[0]))
        al = np.power(10, 0.001)
        dl = P_d*al/P_s
        dl1 = P_d1*al/P_s1
        noise = int_track[0,:len(track)]/20
        noise1 = int_track[0,22:len(track)+22]/5
        mix = track + noise
        mix1 = track + noise1
        #int_track, int_sr = read_wave(os.path.join(home_dir,file))
        '''
        # show results
        plt.figure(figsize=(15, 9))
        plt.subplot(211);
        plt.title("Adaptation");
        plt.xlabel("samples - k")
        plt.plot(mix, 'b--o', label="d - target")
        plt.plot(mix1, 'g--', label="y - output");
        plt.legend()
        plt.subplot(212);
        plt.title("Filter error");
        plt.xlabel("samples - k")
        plt.plot(noise1, "r", label="e - error [dB]");
        plt.plot(noise, "g--", label="e - error [dB]");
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        '''
        sample_lms_rt(mix1, noise, analysis_path, filt_bins)
        #sample_nlms_rt(mix1, noise, analysis_path, filt_bins)
    print("Processing observations...")
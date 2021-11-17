import matplotlib.pylab as plt
import padasip as pa
import numpy as np
import os
from scipy.io import wavfile
from scipy.fftpack import fft
import tensorflow as tf
import soundfile as sf
from tensorflow.python.ops.signal import window_ops
import functools
from tqdm import tqdm
import librosa

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

def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def zs(a):
    """ 1d data z-score """
    a -= a.mean()
    return a / a.std()

def nlms_block(d, n=300, D=200):
    x = pa.input_from_history(d, n)[:-D]
    d = d[n+D-1:]
    #y = y[n+D-1:]
    #q = q[n+D-1:]

    # create filter and filter
    f = pa.filters.FilterNLMS(n=n, mu=0.01, w="zeros")
    yp, e, w = f.run(d, x)
    return yp,e,w
if __name__ == '__main__':

    noisy_dirs = [

        '/home/devpath/datasets/noisy/music'
    ]
    home_dir = '/home/devpath/datasets/anc_enhanced'
    ensures_dir(home_dir)
    #t_polar = tensor_polar(400, 160, 512, 16000)
    for noisy_dir in noisy_dirs:

        w_list = os.listdir(noisy_dir)
        sub_dir = noisy_dir.split('/')[-1]
        tmp_dir = os.path.join(home_dir, sub_dir)
        ensures_dir(tmp_dir)
        print('############################ process type: ' + sub_dir)
        num_labels = len(w_list)
        for idx, f in tqdm(enumerate(w_list), desc='test', total=num_labels):
            # prepare data for simulation
            w_f = os.path.join(noisy_dir, f)
            wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
            yp, e, w = nlms_block(wav_r)
            path1 = os.path.join(tmp_dir, f)
            sf.write(path1, wav_r, 16000)
            path1 = os.path.join(tmp_dir, 'nlms_predict_' + f)
            sf.write(path1, e, 16000)
            '''
            plt.figure(figsize=(15, 9))
            plt.subplot(311)
            plt.title("Adaptation")
            plt.xlabel("samples - k")
            plt.plot(yp, "b", label="predict - target")
            # plt.plot(x0, "g", label="x - input")
            plt.legend()
            plt.subplot(312)
            plt.title("Filter error")
            plt.xlabel("samples - k")
            # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
            plt.plot(wav_r, "g", label="noisy - input")
            # plt.plot(d_t, "b", label="d - input")
            plt.legend()
            plt.legend()
            plt.subplot(313)
            plt.title("Filter error")
            plt.xlabel("samples - k")
            # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
            plt.plot(e, "g", label="error between predict and input")
            # plt.plot(d_t, "b", label="d - input")
            plt.legend()
            plt.tight_layout()
            plt.show()
            #analysis_path = os.path.join(tmp_dir,f)
            path1 = os.path.join(tmp_dir, 'nlms_predict_'+ f)
            sf.write(path1, e, 16000)
            path1 = os.path.join(tmp_dir, 'noisy_' + f)
            sf.write(path1, wav_r, 16000)
            '''
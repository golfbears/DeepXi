import tensorflow as tf
import numpy as np
from scipy import stats
import decimal, math
import os, sys
import librosa
import soundfile as sf
from tensorflow.python.ops.signal import window_ops
import functools
import matplotlib.pyplot as plt
from matplotlib import style
from hybrid.phoneme import Phoneme
from tqdm import tqdm
from histogram2quantile.histogram import histogram
#from mcra.mcra123 import tensor_polar
from pathlib import Path
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


def gaussian_plot():
    style.use('fivethirtyeight')
    mu_params = [-1, 0, 1]
    sd_params = [0.5, 1, 1.5]
    x = np.linspace(-7, 7, 100)
    f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(12, 8))
    for i in range(3):
        for j in range(3):
            mu = mu_params[i]
            sd = sd_params[j]
            y = stats.norm(mu, sd).pdf(x)
            ax[i, j].plot(x, y)
            ax[i, j].plot(0, 0, label='mu={:3.2f}\nsigma={:3.2f}'.format(mu, sd), alpha=0)
            ax[i, j].legend(fontsize=10)
    ax[2, 1].set_xlabel('x', fontsize=16)
    ax[1, 0].set_ylabel('pdf(x)', fontsize=16)
    plt.suptitle('Gaussian PDF', fontsize=16)
    plt.tight_layout()
    plt.show()


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    numSamples = len(signal) - 1
    for i in range(numSamples, 0, -1):
        signal[i] -= coeff * signal[i - 1]

    signal[0] *= (1 - coeff);

    return signal


def framesig(sig, preemph, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.floor((1.0 * slen - frame_len) / frame_step))

    expandedSignal = []
    for i in range(0, numframes):
        for j in range(0, frame_len):
            expandedSignal.append(sig[i * frame_step + j])

    preemphSignal = []
    for i in range(numframes, 0, -1):
        s = (i - 1) * frame_len
        e = i * frame_len
        # preemphSignal.insert(0, preemphasis(expandedSignal[s:e], preemph))
        preemphSignal.insert(0, expandedSignal[s:e])

    res = np.asarray(preemphSignal).flatten()
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) \
              + np.tile(np.arange(0, numframes * frame_len, frame_len), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = res[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


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
        winfunc=lambda x: np.ones((x,))
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
    highfreq = highfreq or samplerate / 2
    frames = framesig(signal, preemph, winlen * samplerate, winstep * samplerate, winfunc)
    energy = np.sum(np.square(frames), 1)  # this stores the total energy in each frame
    energy = np.where(energy == 0, np.finfo(float).eps, energy)  # if energy is zero, we get problems with log

    complex_spec = np.fft.rfft(frames, nfft)

    return np.absolute(complex_spec), np.angle(complex_spec)


def polar_synthesis_np(STMS, STPS, N_d, N_s):
    """
    Polar-form acoustic-domain synthesis.

    Argument/s:
        STMS - short-time magnitude spectrum.
        STPS - short-time phase spectrum.

    Returns:
        Waveform.
    """
    STFT2 = STMS * np.exp(1j * STPS)
    STFT2 = STFT2.transpose()
    synthesis_wav2 = librosa.istft(STFT2, hop_length=N_s, win_length=N_d, window='hamm')

    return synthesis_wav2


class hybridMixMax():
    def __init__(self, x_mu, x_sigma, g_mu, g_sigma, prio_p, alpha, betta):
        self.x_mu = x_mu
        self.x_sigma = x_sigma
        self.g_mu = g_mu
        self.g_sigma = g_sigma
        self.prio_p = prio_p
        self.alpha = alpha
        self.betta = betta
        low_fr = np.ones(185, float) * 0.03
        hi_fr = np.ones(72, float) * 0.01
        self.delta = np.log(np.concatenate((low_fr, hi_fr)))
        self.mixmax_win = np.zeros((4, 257), float)
        self.win_cnt = 0
        self.minimum_pro = 1.0e-10
        self.alpha_d = 0.89

    def gaussian_pdf_scipy(self, mu, sigma, x):
        lengthOfpho = mu.shape[0]
        out = np.stack([stats.norm.pdf(x, mu[i], sigma[i]) for i in range(lengthOfpho)])
        return out

    def gaussian_cdf_scipy(self, mu, sigma, x):
        lengthOfpho = mu.shape[0]
        out = np.stack([stats.norm.cdf(x, mu[i], sigma[i]) for i in range(lengthOfpho)])
        return out

    def f_i_k(self, z):
        return self.gaussian_pdf_scipy(self.x_mu, self.x_sigma, z)

    def g_k(self, z):
        return self.gaussian_pdf_scipy(self.g_mu, self.g_sigma, z)

    def F_i_k(self, z):
        return self.gaussian_cdf_scipy(self.x_mu, self.x_sigma, z)

    def G_k(self, z):
        return self.gaussian_cdf_scipy(self.g_mu, self.g_sigma, z)

    def f_G_k(self, z):
        f_i_k = self.f_i_k(z)
        G_k = self.G_k(z)
        o = f_i_k * G_k
        return o

    def F_g_k(self, z):
        o = self.F_i_k(z) * self.g_k(z)
        return o

    def h_i_k(self, z):
        o = self.f_i_k(z) * self.G_k(z) + self.F_i_k(z) * self.g_k(z)
        return o

    def rho_i_k(self, z):
        f_G_k = self.f_G_k(z)
        F_g_k = self.F_g_k(z)
        f_G_k = np.where(f_G_k == 0, np.finfo(float).eps, f_G_k)
        F_g_k = np.where(F_g_k == 0, np.finfo(float).eps, F_g_k)
        o = f_G_k / (f_G_k + F_g_k)
        return o

    def rho_i_k_and_h_i_k(self, p_pre, z):
        f_i_k = self.f_i_k(z)
        G_k = self.G_k(z)
        F_i_k = self.F_i_k(z)
        g_k = self.g_k(z)
        h_i_k = f_i_k * G_k + F_i_k * g_k
        # h_i = np.prod(h_i_k, axis=1)
        log_h_i = np.sum(np.log(h_i_k), axis=1)
        h_i = np.exp(log_h_i)
        h = np.sum(p_pre * h_i, axis=0)
        h = np.where(h == 0, np.finfo(float).eps, h)
        p_mm = p_pre * h_i / h
        f_G_k = f_i_k * G_k
        F_g_k = F_i_k * g_k
        F_g_k = np.where(F_g_k == 0, np.finfo(float).eps, F_g_k)
        rho_i_k = f_G_k / (f_G_k + F_g_k)
        o = np.expand_dims(p_mm, 1) * rho_i_k
        return np.sum(o, axis=0), p_mm

    def rho_i_k_mixmax(self, p_pre, z):
        f_i_k = self.f_i_k(z)
        G_k = self.G_k(z)
        F_i_k = self.F_i_k(z)
        g_k = self.g_k(z)
        h_i_k = f_i_k * G_k + F_i_k * g_k
        h_i_k = np.where(h_i_k == 0, np.finfo(float).eps, h_i_k)
        # h_i_k = np.maximum(h_i_k, self.minimum_pro)
        #h_i = np.prod(h_i_k, axis=1)
        log_h_i = np.sum(np.log(h_i_k), axis=1)
        h_i = np.exp(log_h_i)
        h_i = np.where(h_i == 0, np.finfo(float).eps, h_i)
        h = np.sum(p_pre * h_i, axis=0)
        h = np.where(h == 0, np.finfo(float).eps, h)
        p_mm = p_pre * h_i / h
        f_G_k = f_i_k * G_k
        F_g_k = F_i_k * g_k
        F_g_k = np.where(F_g_k == 0, np.finfo(float).eps, F_g_k)
        rho_i_k = f_G_k / (f_G_k + F_g_k)

        F_i_k = np.where(F_i_k == 0, np.finfo(float).eps, F_i_k)
        R_i_k = f_i_k / F_i_k
        return rho_i_k, R_i_k, p_mm

    def rho_NN_MM(self, p_NN, z):
        o = np.expand_dims(p_NN, 1) * self.rho_i_k(z)
        o = np.sum(o, axis=0)
        return o

    def tracking_mu_sigma(self, z, rho):
        self.g_mu = rho * self.g_mu + (1 - rho) * (self.alpha * z + (1 - self.alpha) * self.g_mu)
        self.g_sigma = rho * self.g_sigma + (1 - rho) * (
                self.alpha * np.sqrt(np.square(z - self.g_mu)) + (1 - self.alpha) * self.g_sigma)
        self.g_sigma = np.where(self.g_sigma == 0, np.finfo(float).eps, self.g_sigma)

    def x_estimate_mixmax_bak(self, p_prev, z):
        rho_nn_mm, R_i_k, p_mm = self.rho_i_k_mixmax(p_prev, z)
        p_mm = np.where(p_mm == 0, np.finfo(float).eps, p_mm)
        rho_nn_mm_a = np.sum(np.expand_dims(p_mm, 1) * rho_nn_mm, axis=0)
        mixmax_betta = self.x_mu - np.square(self.x_sigma) * R_i_k
        # mixmax_betta = self.x_mu - self.x_sigma * R_i_k
        mixmax_betta_a = np.sum(mixmax_betta, axis=0)
        # a1 = np.minimum(z*0.95,mixmax_betta_a)
        # mixmax_betta = self.update_mixmax_win(a1)
        mixmax_betta = mixmax_betta_a
        o = rho_nn_mm_a * z + mixmax_betta * (1 - rho_nn_mm_a)
        o = np.max((z + self.delta, o), axis=0)

        return o, rho_nn_mm_a

    def x_estimate_mixmax(self, p_prev, z):
        # rho_nn_mm, R_i_k, p_mm = self.rho_i_k_mixmax(p_prev,z)
        rho_nn_mm, R_i_k, p_mm = self.rho_i_k_mixmax(self.prio_p, z)
        # p_mm = np.where(p_mm == 0, np.finfo(float).eps, p_mm)
        mixmax_betta = self.x_mu - np.square(self.x_sigma) * R_i_k
        # mixmax_betta = self.x_mu - self.x_sigma * R_i_k

        o = rho_nn_mm * np.expand_dims(z, 0) + mixmax_betta * (1 - rho_nn_mm)
        o = np.sum(np.expand_dims(p_mm, 1) * o, axis=0)
        o = np.max((z + self.delta, o), axis=0)

        # rho_nn_mm = np.sum(np.expand_dims(p_prev, 1)*rho_nn_mm, axis=0)
        rho_nn_mm = np.sum(np.expand_dims(p_prev, 1) * rho_nn_mm, axis=0)
        rho_nn_mm = np.where(rho_nn_mm == 0, np.finfo(float).eps, rho_nn_mm)
        # rho_nn_mm = np.sum(rho_nn_mm, axis=0)

        return o, rho_nn_mm

    def x_estimate_mixmax_nn(self, p_prev, z):
        rho_nn_mm, R_i_k, p_mm = self.rho_i_k_mixmax(p_prev, z)
        mixmax_betta = self.x_mu - np.square(self.x_sigma) * R_i_k
        # mixmax_betta = self.x_mu-(self.x_sigma)*R_i_k
        # mixmax_betta_a = np.max((self.g_mu*(1-rho_nn_mm), mixmax_betta), axis=0)
        o = rho_nn_mm * np.expand_dims(z, 0) + mixmax_betta * (1 - rho_nn_mm)
        o = np.sum(np.expand_dims(p_mm, 1) * o, axis=0)
        o = np.max((z + self.delta, o), axis=0)

        rho_nn_mm = np.sum(np.expand_dims(p_mm, 1) * rho_nn_mm, axis=0)
        # rho_nn_mm = np.sum(rho_nn_mm, axis=0)

        return o, rho_nn_mm

    def x_estimate_mm(self, p_prev, z):
        rho_nn_mm, p_mm = self.rho_i_k_and_h_i_k(p_prev, z)
        o = z - (1 - rho_nn_mm) * self.betta
        return o, p_mm

    def x_estimate(self, p_NN, z):
        rho_nn_mm = self.rho_NN_MM(p_NN, z)
        o = z - (1 - rho_nn_mm) * self.betta

        return o

    def updata_alpha_betta(self, a, b):
        self.alpha = a
        self.betta = b

    def update_noise(self, mu, sigma):
        self.g_mu = mu
        self.g_sigma = sigma

    def get_noise(self):
        return self.g_mu, self.g_sigma

    def init_mixmax_win(self, z):
        self.mixmax_win = self.mixmax_win + np.expand_dims(z, 0)

    def update_mixmax_win(self, z):
        self.mixmax_win[self.win_cnt] = z
        self.win_cnt = (self.win_cnt + 1) % 3
        return np.mean(self.mixmax_win, axis=0)


def simple_extract_gaussians():
    phonemes_dict_dir = '/home/devpath/golfbears/speech-aligner/egs/cn_phn/res/phones.txt'
    with open(phonemes_dict_dir, 'r') as f_phonemes_dict:
        phonemes_dict = f_phonemes_dict.readlines()[1:201]
    phonemes_dir = '/home/devpath/ml/speech-aligner/egs/cn_phn/data/analysis_MoG/gaussian_shellttd_aidatattd_thchs30'

    f_list = os.listdir(phonemes_dir)
    means = []
    stds = []
    probs = []
    apps = []
    phonenemes_dict = {}
    for i, phoneme in enumerate(phonemes_dict):
        if phoneme.__contains__('sil') or phoneme.__contains__('$0'):
            # if phoneme.__contains__('$0'):
            continue
        pho = phoneme.split(' ')[0]
        print('############################ check phoneme: No.' + str(i) + '-' + pho)
        available = 0
        for file in f_list:
            if not file.__contains__(pho):
                continue
            available = 1
            file_dir = os.path.join(phonemes_dir, file)
            if file.__contains__('mean'):
                mean = np.load(file_dir)
            elif file.__contains__('std'):
                std = np.load(file_dir)
            elif file.__contains__('probility'):
                pro = np.load(file_dir)
            elif file.__contains__('appears'):
                app = np.load(file_dir)
            else:
                print('not correct phoneme: No.' + str(i) + '-' + pho)
        if available == 0:
            print('not available phoneme: No.' + str(i) + '-' + pho)
            continue
        means.append(mean)
        stds.append(std)
        probs.append(pro)
        apps.append(app)
    return means, stds, probs


def phoneme_extract_gaussians():
    ph_label_mapper = Phoneme(
        os.path.join('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt'))

    # phonemes_dir = '/home/devpath/rtx3080/ml/speech-aligner/egs/cn_phn/data/aidatatang/phonemes_gaussians_calculate_epoch1000'
    phonemes_dir = '/home/devpath/rtx3080/ml/speech-aligner/egs/cn_phn/data/shell/phonemes_gaussians_calculate_epoch616_1'
    # phonemes_dir = '/home/devpath/golfbears/mfb/mfb_gaussCalc'
    f_list = os.listdir(phonemes_dir)
    means = []
    stds = []
    probs = []
    apps = []
    i = 0
    for pho in ph_label_mapper.tkn_dict.entry2Index:
        print(pho + ':' + str(int(ph_label_mapper.tkn_dict.entry2Index[pho])))
        if pho.__contains__('|') or pho.__contains__('*') or pho.__contains__('er1'):
            continue
        print('############################ check phoneme: No.' + str(i) + '-' + pho)

        available = 0
        for file in f_list:
            if not file.__contains__(pho):
                continue
            available = 1
            file_dir = os.path.join(phonemes_dir, file)
            if file.__contains__('mean'):
                mean = np.load(file_dir)
            elif file.__contains__('std'):
                std = np.load(file_dir)
            elif file.__contains__('probility'):
                pro = np.load(file_dir)
            elif file.__contains__('appears'):
                app = np.load(file_dir)
            else:
                print('not correct phoneme: No.' + str(i) + '-' + pho)
        if available == 0:
            print('not available phoneme: No.' + str(i) + '-' + pho)
            continue
        i += 1
        means.append(mean)
        stds.append(std)
        probs.append(pro)
        apps.append(app)
    return means, stds, probs


def phoneme_mfa_gaussians():
    phonemes_dir = '/home/devpath/rtx3080/ml/mfa_gauss'
    ali = os.path.join(phonemes_dir, 'dict.txt')
    with open(ali, 'r') as f_ali:
        for i, a_line in enumerate(f_ali):
            print(a_line + ':' + str(i))
    means = np.load(os.path.join(phonemes_dir, 'meanMfa.npy'))
    stds = np.load(os.path.join(phonemes_dir, 'stdMfa.npy'))
    probs = np.ones(stds.shape[0]) / stds.shape[0]

    return means, stds, probs


def phoneme_lmfb_gaussians():
    phonemes_dir = '/home/devpath/golfbears/mfb/mfb_gaussCalc'

    f_list = os.listdir(phonemes_dir)
    pho2index = []
    for f in f_list:
        phon = f.split('_')[0]
        if not pho2index.__contains__(phon):
            pho2index.append(phon)
    means = []
    stds = []
    probs = []
    apps = []
    i = 0
    for kkk, pho in enumerate(pho2index):
        print(pho + ':' + str(kkk))
        if pho.__contains__('|') or pho.__contains__('*') or pho.__contains__('er1'):
            continue
        print('############################ check phoneme: No.' + str(i) + '-' + pho)

        available = 0
        for file in f_list:
            if not file.__contains__(pho):
                continue
            available = 1
            file_dir = os.path.join(phonemes_dir, file)
            if file.__contains__('mean'):
                mean = np.load(file_dir)
            elif file.__contains__('std'):
                std = np.load(file_dir)
            elif file.__contains__('probility'):
                pro = np.load(file_dir)
            elif file.__contains__('appears'):
                app = np.load(file_dir)
            else:
                print('not correct phoneme: No.' + str(i) + '-' + pho)
        if available == 0:
            print('not available phoneme: No.' + str(i) + '-' + pho)
            continue
        i += 1
        means.append(mean)
        stds.append(std)
        probs.append(pro)
        apps.append(app)
    return means, stds, probs

def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)
if __name__ == '__main__':
    means, stds, probs = phoneme_mfa_gaussians()
    means = means[1:]
    stds = stds[1:]
    probs = probs[1:]
    # means1, stds1, probs1 = phoneme_extract_gaussians()

    # means, stds, probs = simple_extract_gaussians()
    home_dir = '/home/devpath/datasets/mfa_gauss/enhanced_old'

    noisy_dirs = [

        '/homedevpath/datasets/speech_corpus/train_BF/configfourmic_BF'
    ]

    probs_dirs = '/home/devpath/datasets/speech_corpus/mfa_probs'
    t_polar = tensor_polar(400, 160, 512, 16000)
    for noisy_dir in noisy_dirs:
        w_list = os.listdir(noisy_dir)
        sub_dir = noisy_dir.split('/')[-1]
        print('############################ process type: ' + sub_dir)
        prob_d = os.path.join(probs_dirs, sub_dir)
        path_tmp = os.path.join(home_dir, sub_dir)
        ensures_dir(path_tmp)
        num_labels = len(w_list)
        #for f in w_list:from tqdm import tqdm
        for idx, f in tqdm(enumerate(w_list), desc='test', total=num_labels):
            prob_f = os.path.join(prob_d, f.split('.')[0] + '_prob.npy')
            if not os.path.isfile(prob_f):
                continue
            prob_nn = np.load(prob_f)

            w_f = os.path.join(noisy_dir, f)
            wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
            meg1, angle1 = t_polar.polar_analysis(wav_r)
            meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
            logmeg1 = np.log(meg1)
            alpha_list = np.arange(0.007, 0.008, 0.001)
            betta_list = np.arange(0.5, 0.7, 0.2)
            mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
            stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
            h_m_max = hybridMixMax(np.array(means), np.array(stds), mean, stand, np.array(probs), 0.2, 1.2)
            g_hists = histogram(logmeg1[0], alpha_d=0.9, alpha_s=0.9, frame_L=100, fft_len=512,
                                delta=5)
            fr_length = np.minimum(prob_nn.shape[0],logmeg1.shape[0])
            for a in alpha_list:
                for b in betta_list:

                    h_m_max.updata_alpha_betta(a, b)
                    h_m_max.update_noise(mean, stand)
                    #print('############################ process hyperparams: alpha.' + str(round(a, 1)) + ' betta.' + str(round(b, 1)))

                    # x_hat = np.vstack([h_m_max.x_estimate_mm(np.array(probs), logf) for logf in logmeg1])
                    # meg = np.exp(x_hat)
                    x_hat = np.zeros(257, float)
                    for i, logf in enumerate(logmeg1[:fr_length]):
                        #hists_noise, hists_noise1, hists_noise2 = g_hists.tracking_noise(logf, i)
                        #h_n_mu, h_n_std, h_n1_mu, h_n1_std, h_n2_mu, h_n2_std = g_hists.get_mu_std()
                        #h_m_max.update_noise(np.expand_dims(h_n2_mu, 0), np.expand_dims(h_n2_std, 0))
                        o, rtho = h_m_max.x_estimate_mixmax_nn(prob_nn[i,1:], logf)
                        h_m_max.tracking_mu_sigma(o,rtho)
                        # o = h_m_max.x_estimate_mm(np.array(probs), logf)
                        x_hat = np.vstack((x_hat, o))
                    meg = np.exp(x_hat[1:])
                    wav2 = t_polar.polar_synthesis(meg[:fr_length], angle1[:fr_length])
                    wav2 = np.squeeze(wav2)

                    path2 = os.path.join(path_tmp, f)
                    sf.write(path2, wav2, 16000)

    '''                
    home_dir = '/home/devpath/datasets/hybridwav/npy'
    w_f = os.path.join(home_dir, 'gmm04_001_00-SNR0DB.wav')
    wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
    meg1, angle1 = stft_analysis(wav_r, samplerate=16000,
                     winlen=0.025,
                     winstep=0.01
                     )
    meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
    logmeg1 = np.log(meg1)
    alpha_list = np.arange(0.4, 1.0, 0.1)
    betta_list = np.arange(0.5, 2.5, 0.1)
    mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
    stand =np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
    h_m_max = hybridMixMax(np.array(means), np.array(stds), mean, stand, np.array(probs), 0.2, 1.2)
    prob_nn = np.load(os.path.join(home_dir,'gmm04_001_00-SNR0DB_prob.npy'))
    for a in alpha_list:
        for b in betta_list:

            h_m_max.updata_alpha_betta(a, b)
            h_m_max.update_noise(mean, stand)
            print('############################ process hyperparams: alpha.' + str(round(a,1)) + ' betta.'+ str(round(b,1)))

            #x_hat = np.vstack([h_m_max.x_estimate_mm(np.array(probs), logf) for logf in logmeg1])
            #meg = np.exp(x_hat)
            x_hat = np.zeros(257, float)
            for i, logf in enumerate(logmeg1):
                #o = h_m_max.x_estimate(prob_nn[i], logf)
                o = h_m_max.x_estimate_mm(np.array(probs), logf)
                x_hat = np.vstack((x_hat, o))
            meg = np.exp(x_hat[1:])
            wav2 = polar_synthesis_np(meg, angle1, 400,160)
            wav2 = np.squeeze(wav2)
            path2 = os.path.join(home_dir, 'gmm_gen/gmm04_001_00-SNR0DB_'+str(round(a,1))+'_'+str(round(b,1))+'.wav')
            sf.write(path2, wav2, 16000)
    '''

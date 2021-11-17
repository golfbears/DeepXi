import tensorflow as tf
import numpy as np
from scipy import stats
import decimal, math
import os, sys
import librosa
import soundfile as sf

import matplotlib.pyplot as plt
from matplotlib import style
from hybrid.phoneme import Phoneme
from hybrid.hybridMixMax import hybridMixMax,simple_extract_gaussians, phoneme_extract_gaussians
from asr_mfcc.base import logfbank
from mcra.mcra123 import mcra, mcra_2, imcra

def maptoheat1(x, cmap_style):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 64
    x += 128
    x = np.clip(x, 0, 255).astype('uint8')
    plt.figure()
    plt.imshow(x, aspect='auto', cmap=cmap_style, origin='lower')
    plt.show()

def maptoheat2(x, cmap_style):

    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 128
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def maptoheat3(x, cmap_style):
    x = np.maximum(x, 0)
    x /= np.max(x)+ 1e-5

    #plt.figure()
    plt.matshow(x, aspect='auto', cmap=cmap_style, origin='lower')
    #plt.show()

def gaussian_plot():
    style.use('fivethirtyeight')
    mu_params = [-1, 0, 1]
    sd_params = [0.5, 1, 1.5]
    x = np.linspace(-7, 7, 100)
    f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(12,8))
    for i in range(3):
        for j in range(3):
            mu = mu_params[i]
            sd = sd_params[j]
            y = stats.norm(mu, sd).pdf(x)
            ax[i, j].plot(x, y)
            ax[i, j].plot(0,0, label='mu={:3.2f}\nsigma={:3.2f}'.format(mu,sd), alpha=0)
            ax[i, j].legend(fontsize=10)
    ax[2,1].set_xlabel('x', fontsize=16)
    ax[1,0].set_ylabel('pdf(x)', fontsize=16)
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
    highfreq= highfreq or samplerate/2
    frames = framesig(signal, preemph, winlen * samplerate, winstep * samplerate, winfunc)
    energy = np.sum(np.square(frames),1) # this stores the total energy in each frame
    energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log


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


class gmm_phoneme_noise():
    def __init__(self, x_mu, x_sigma, prio_p, g_mu, g_sigma, n_gmm_dimesion):
        """
        :param x_mu: input is the average of log(Amp), *2 means the average of log(Amp*Amp) = 2*log(Amp) => meanPwr = 2* meanAmp
        :param x_sigma: input is the std of log(Amp), * means [sum(2*log(Amp-i)-meanPowr]/N => stdPwr= 2* stdAmp
        :param prio_p:
        :param g_mu:
        :param g_sigma:
        :param n_gmm_dimesion:
        :param input_x:
        """

        pho_gmm_dimension = np.shape(x_mu)[1]
        self.x_mu = np.expand_dims(x_mu, 0)
        self.x_Sigma = np.expand_dims(x_sigma, 0)
        self.x_Sigma_recip = np.reciprocal(self.x_Sigma)
        self.w_sk = np.expand_dims(prio_p, 0)
        #self.g_mu = np.expand_dims(np.tile(g_mu,n_gmm_dimesion).reshape([n_gmm_dimesion,-1]),1)
        rand_noise = np.random.rand(n_gmm_dimesion, pho_gmm_dimension) * np.tile(g_mu,n_gmm_dimesion).reshape([n_gmm_dimesion,-1])
        std_noise = np.sqrt(g_sigma)

        self.g_mu = np.expand_dims(np.stack(
                        [np.stack(
                            [stats.norm.pdf(rand_noise[j, i], g_mu[0,i], std_noise[0,i]) for i in range(rand_noise.shape[1])]
                                ) for j in range(rand_noise.shape[0])]
                         ), 1)
        self.g_Sigma = np.expand_dims(np.tile(g_sigma,n_gmm_dimesion).reshape([n_gmm_dimesion,-1]),1)

        self.w_nl = np.expand_dims(np.ones(n_gmm_dimesion, float)/n_gmm_dimesion,1)
        self.bias = np.expand_dims(np.zeros(pho_gmm_dimension, float), 0)
        llow_fr = np.ones(7, float) * 0.3
        low_fr = np.ones(10, float) * 0.5
        mid_fr = np.ones(48, float) * 0.65
        hi_fr = np.ones(192, float) * 0.78
        self.alpha_d = np.expand_dims(np.concatenate((llow_fr, low_fr, mid_fr, hi_fr)),0)
        #self.alpha_d = 0.79


    def get_weight_o_j_k_l(self):
        w = self.w_sk * self.w_nl
        return w
    def get_h_mu_jacobian(self):
        e_bias = np.expand_dims(self.bias, 0)
        mu_delta = np.exp(self.g_mu - self.x_mu - e_bias)
        mismatch_add_1 = 1 + mu_delta
        mismatch = np.log(mismatch_add_1)
        self.mismatch_signal = mismatch + e_bias
        self.mismatch_signal = np.where(self.mismatch_signal == 0, np.finfo(float).eps, self.mismatch_signal)
        mu_o = self.x_mu + mismatch + e_bias
        derivatives = 1 - 1/mismatch_add_1
        mismatch_n = np.reciprocal(mu_delta)
        self.mismatch_noise = np.log(mismatch_n + 1)
        return mu_o, derivatives

    def get_g_sigma_o_j_k_l(self, h_mu_jacobians):
        delta_jacobians = 1 - h_mu_jacobians
        Sigma_s_jac = delta_jacobians * self.x_Sigma * delta_jacobians
        Sigma_n_jac = h_mu_jacobians * self.g_Sigma * h_mu_jacobians
        Sigma_o = Sigma_s_jac + Sigma_n_jac
        return Sigma_o

    def get_h_mu_jacobian_2(self):
        e_bias = np.expand_dims(self.bias, 0)
        mu_delta = np.exp(self.g_mu - self.x_mu - e_bias)
        mismatch_add_1 = 1 + mu_delta
        mismatch = np.log(mismatch_add_1)
        mu_o = self.x_mu + mismatch + e_bias
        derivatives = 1/mismatch_add_1
        return mu_o, derivatives

    def get_g_sigma_o_j_k_l_2(self, h_mu_jacobians):
        delta_jacobians = 1 - h_mu_jacobians
        Sigma_n_jac = delta_jacobians * self.g_Sigma * delta_jacobians
        Sigma_s_jac = h_mu_jacobians * self.x_Sigma * h_mu_jacobians
        Sigma_o = Sigma_s_jac + Sigma_n_jac
        return Sigma_o

    def calculate_P_o_t_k_l(self, x):

        std = np.sqrt(self.Sigma_okl)

        pdf = np.stack(
                [np.stack(
                    [np.stack(
                        [stats.norm.pdf(x[k], self.mu_okl[j, i], std[j, i]) for i in range(self.mu_okl.shape[1])]
                             ) for j in range(self.mu_okl.shape[0])]
                         )for k in range(x.shape[0])]
                      )
        gmm_pdf = np.product(pdf, -1)
        gmm_pdf = np.where(gmm_pdf == 0, np.finfo(float).eps, gmm_pdf)
        '''
        gmm_pdf = np.stack(
            [np.stack(
                [np.stack(
                    [stats.multivariate_normal.pdf(x[k], self.mu_okl[j, i], self.Sigma_okl[j, i]) for i in range(self.mu_okl.shape[1])]
                ) for j in range(self.mu_okl.shape[0])]
            ) for k in range(x.shape[0])]
        )
        '''
        p_init = np.expand_dims(self.w_okl, 0) * gmm_pdf

        normalized_denominator = np.sum(np.sum(p_init, axis=1), 1)
        normalized_denominator = np.where(normalized_denominator == 0, np.finfo(float).eps, normalized_denominator)
        self.p_o_t_k_l = p_init/np.expand_dims(np.expand_dims(normalized_denominator,1), 1)
        self.p_stk = np.sum(self.p_o_t_k_l, axis=1)
        self.p_ntl = np.sum(self.p_o_t_k_l, axis=2)

    def hybrid_P_o_t_k_l(self, x, stk_nn):
        self.calculate_P_o_t_k_l(x)
        self.p_stk = stk_nn
        #self.p_stk = (stk_nn + self.p_stk) / 2
        self.p_o_t_k_l = np.expand_dims(self.p_ntl, axis=2) * np.expand_dims(stk_nn, axis=1)


    def update_signal_noise(self, x):
        e_bias = np.expand_dims(self.bias, 0)
        Signal_hat = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.x_mu+e_bias-self.mu_okl, 0)
        Signal_hat = x + np.sum(np.sum(Signal_hat,axis=1), axis=1)
        Noise_hat = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.g_mu-self.mu_okl, 0)
        Noise_hat = x + np.sum(np.sum(Noise_hat, axis=1), axis=1)
        Noise_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_noise, 0)
        self.Noise_hat2 = x - np.sum(np.sum(Noise_hat2, axis=1), axis=1)
        Noise_hat = self.Noise_hat2
        n_gmm = np.sum(self.p_ntl, 0)
        self.w_nl = np.expand_dims(n_gmm/np.expand_dims(np.sum(n_gmm, 0),0),1)
        self.g_mu = np.expand_dims(np.sum(np.expand_dims(self.p_ntl, -1)*np.expand_dims(Noise_hat,1), 0)/np.expand_dims(n_gmm,-1),1)
        dela = np.sum(np.expand_dims(self.p_ntl, -1)*np.expand_dims(Noise_hat*Noise_hat,1), 0)
        g_Sigma = np.expand_dims(dela/np.expand_dims(n_gmm,-1),1) - self.g_mu*self.g_mu
        g_Sigma = np.where(g_Sigma == 0, np.finfo(float).eps, g_Sigma)

        #sig_square = np.sum(np.expand_dims(np.expand_dims(self.p_ntl, -1),2)*np.square(np.expand_dims(np.expand_dims(Noise_hat,1),1) - np.expand_dims(self.g_mu,0)), 0)
        #self.g_Sigma = sig_square/np.expand_dims(n_gmm,1)

        self.g_Sigma = g_Sigma
        s_pro = np.expand_dims(self.p_stk, -1)*np.expand_dims(self.x_Sigma_recip,0)

        s_pro = np.where(s_pro == 0, np.finfo(float).eps, s_pro)
        b_bias = s_pro*(np.expand_dims(x,1)-self.x_mu)
        self.bias = np.reciprocal(np.sum(np.sum(s_pro,1),1)) * np.sum(np.sum(b_bias,1),1)
        return Signal_hat, Noise_hat#, self.bias

    def update_signal_noise_spp(self, x, spp):
        sap = 1 - spp
        e_bias = np.expand_dims(self.bias, 0)
        #alpha_d_hat = self.alpha_d + (1-self.alpha_d)*spp
        Signal_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_signal, 0)
        Signal_hat2 = x - (sap*x + spp*(np.sum(np.sum(Signal_hat2, axis=1), axis=1)))*self.alpha_d
        #Signal_hat2 = x - alpha_d_hat*x - (1-alpha_d_hat)*(np.sum(np.sum(Signal_hat2, axis=1), axis=1))
        Noise_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_noise, 0)
        #Noise_hat2 = sap*x + spp*(x - np.sum(np.sum(Noise_hat2, axis=1), axis=1))
        Noise_hat2 = x - spp * np.sum(np.sum(Noise_hat2, axis=1), axis=1)

        n_gmm = np.sum(self.p_ntl, 0)
        self.w_nl = np.expand_dims(n_gmm/np.expand_dims(np.sum(n_gmm, 0),0),1)
        self.g_mu = np.expand_dims(np.sum(np.expand_dims(self.p_ntl, -1)*np.expand_dims(Noise_hat2,1), 0)/np.expand_dims(n_gmm,-1),1)
        dela = np.sum(np.expand_dims(self.p_ntl, -1)*np.expand_dims(Noise_hat2*Noise_hat2,1), 0)
        g_Sigma = np.expand_dims(dela/np.expand_dims(n_gmm,-1),1) - self.g_mu*self.g_mu
        g_Sigma = np.where(g_Sigma == 0, np.finfo(float).eps, g_Sigma)

        self.g_Sigma = g_Sigma
        s_pro = np.expand_dims(self.p_stk, -1)*np.expand_dims(self.x_Sigma_recip,0)

        s_pro = np.where(s_pro == 0, np.finfo(float).eps, s_pro)
        b_bias = s_pro*(np.expand_dims(x,1)-self.x_mu)
        self.bias = np.reciprocal(np.sum(np.sum(s_pro,1),1)) * np.sum(np.sum(b_bias,1),1)
        return Signal_hat2, Noise_hat2

    def compensate_model(self):
        self.w_okl = self.get_weight_o_j_k_l()
        self.mu_okl, self.jacobians = self.get_h_mu_jacobian()
        self.Sigma_okl = self.get_g_sigma_o_j_k_l(self.jacobians)



class rt_vts_noise():
    def __init__(self, x_mu, x_sigma, prio_p, g_mu, g_sigma, n_gmm_dimesion):
        """
        :param x_mu: input is the average of log(Amp), *2 means the average of log(Amp*Amp) = 2*log(Amp) => meanPwr = 2* meanAmp
        :param x_sigma: input is the std of log(Amp), * means [sum(2*log(Amp-i)-meanPowr]/N => stdPwr= 2* stdAmp
        :param prio_p:
        :param g_mu:
        :param g_sigma:
        :param n_gmm_dimesion:
        :param input_x:
        """

        pho_gmm_dimension = np.shape(x_mu)[1]
        self.x_mu = np.expand_dims(x_mu, 0)
        self.x_Sigma = np.expand_dims(x_sigma, 0)
        self.x_Sigma_recip = np.reciprocal(self.x_Sigma)
        self.w_sk = np.expand_dims(prio_p, 0)
        #self.g_mu = np.expand_dims(np.tile(g_mu,n_gmm_dimesion).reshape([n_gmm_dimesion,-1]),1)
        rand_noise = np.random.rand(n_gmm_dimesion, pho_gmm_dimension) * np.tile(g_mu,n_gmm_dimesion).reshape([n_gmm_dimesion,-1])
        std_noise = np.sqrt(g_sigma)
        self.g_mu = np.expand_dims(np.stack(
            [np.stack(
                [stats.norm.pdf(rand_noise[j, i], g_mu[0, i], std_noise[0, i]) for i in range(rand_noise.shape[1])]
            ) for j in range(rand_noise.shape[0])]
        ), 1)
        self.g_Sigma = np.expand_dims(np.tile(g_sigma, n_gmm_dimesion).reshape([n_gmm_dimesion, -1]), 1)

        self.w_nl = np.expand_dims(np.ones(n_gmm_dimesion, float) / n_gmm_dimesion, 1)
        self.bias = np.expand_dims(np.zeros(pho_gmm_dimension, float), 0)
        llow_fr = np.ones(7, float) * 0.3
        low_fr = np.ones(10, float) * 0.5
        mid_fr = np.ones(48, float) * 0.65
        hi_fr = np.ones(192, float) * 0.78
        self.alpha_d = np.expand_dims(np.concatenate((llow_fr, low_fr, mid_fr, hi_fr)), 0)
        # self.alpha_d = 0.79

    def get_weight_o_j_k_l(self):
        w = self.w_sk * self.w_nl
        return w
    def get_h_mu_jacobian(self):
        e_bias = np.expand_dims(self.bias, 0)
        mu_delta = np.exp(self.g_mu - self.x_mu - e_bias)
        mismatch_add_1 = 1 + mu_delta
        mismatch = np.log(mismatch_add_1)
        self.mismatch_signal = mismatch + e_bias
        self.mismatch_signal = np.where(self.mismatch_signal == 0, np.finfo(float).eps, self.mismatch_signal)
        mu_o = self.x_mu + mismatch + e_bias
        derivatives = 1 - 1/mismatch_add_1
        mismatch_n = np.reciprocal(mu_delta)
        self.mismatch_noise = np.log(mismatch_n + 1)
        return mu_o, derivatives

    def get_g_sigma_o_j_k_l(self, h_mu_jacobians):
        delta_jacobians = 1 - h_mu_jacobians
        Sigma_s_jac = delta_jacobians * self.x_Sigma * delta_jacobians
        Sigma_n_jac = h_mu_jacobians * self.g_Sigma * h_mu_jacobians
        Sigma_o = Sigma_s_jac + Sigma_n_jac
        return Sigma_o

    def get_h_mu_jacobian_2(self):
        e_bias = np.expand_dims(self.bias, 0)
        mu_delta = np.exp(self.g_mu - self.x_mu - e_bias)
        mismatch_add_1 = 1 + mu_delta
        mismatch = np.log(mismatch_add_1)
        mu_o = self.x_mu + mismatch + e_bias
        derivatives = 1/mismatch_add_1
        return mu_o, derivatives

    def get_g_sigma_o_j_k_l_2(self, h_mu_jacobians):
        delta_jacobians = 1 - h_mu_jacobians
        Sigma_n_jac = delta_jacobians * self.g_Sigma * delta_jacobians
        Sigma_s_jac = h_mu_jacobians * self.x_Sigma * h_mu_jacobians
        Sigma_o = Sigma_s_jac + Sigma_n_jac
        return Sigma_o

    def calculate_P_o_t_k_l(self, x):

        std = np.sqrt(self.Sigma_okl)

        pdf = np.stack(
                [np.stack(
                    [np.stack(
                        [stats.norm.pdf(x[k], self.mu_okl[j, i], std[j, i]) for i in range(self.mu_okl.shape[1])]
                             ) for j in range(self.mu_okl.shape[0])]
                         )for k in range(x.shape[0])]
                      )
        gmm_pdf = np.product(pdf, -1)
        gmm_pdf = np.where(gmm_pdf == 0, np.finfo(float).eps, gmm_pdf)
        '''
        gmm_pdf = np.stack(
            [np.stack(
                [np.stack(
                    [stats.multivariate_normal.pdf(x[k], self.mu_okl[j, i], self.Sigma_okl[j, i]) for i in range(self.mu_okl.shape[1])]
                ) for j in range(self.mu_okl.shape[0])]
            ) for k in range(x.shape[0])]
        )
        '''
        p_init = np.expand_dims(self.w_okl, 0) * gmm_pdf

        normalized_denominator = np.sum(np.sum(p_init, axis=1), 1)
        normalized_denominator = np.where(normalized_denominator == 0, np.finfo(float).eps, normalized_denominator)
        self.p_o_t_k_l = p_init/np.expand_dims(np.expand_dims(normalized_denominator,1), 1)
        self.p_stk = np.sum(self.p_o_t_k_l, axis=1)
        self.p_ntl = np.sum(self.p_o_t_k_l, axis=2)

    def hybrid_P_o_t_k_l(self, x, stk_nn):
        self.calculate_P_o_t_k_l(x)
        self.p_stk = stk_nn
        #self.p_stk = (stk_nn + self.p_stk) / 2
        self.p_o_t_k_l = np.expand_dims(self.p_ntl, axis=2) * np.expand_dims(stk_nn, axis=1)

    def compensate_model(self):
        self.w_okl = self.get_weight_o_j_k_l()
        self.mu_okl, self.jacobians = self.get_h_mu_jacobian()
        self.Sigma_okl = self.get_g_sigma_o_j_k_l(self.jacobians)

    def update_mu_sigma(self, mu, sigma):
        self.g_mu = np.expand_dims(mu, 0)
        self.g_Sigma = np.expand_dims(sigma, 0)
    def update_signal_noise(self, x):
        e_bias = np.expand_dims(self.bias, 0)
        Signal_hat = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.x_mu+e_bias-self.mu_okl, 0)
        Signal_hat = x + np.sum(np.sum(Signal_hat,axis=1), axis=1)
        Noise_hat = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.g_mu-self.mu_okl, 0)
        Noise_hat = x + np.sum(np.sum(Noise_hat, axis=1), axis=1)
        Noise_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_noise, 0)
        self.Noise_hat2 = x - np.sum(np.sum(Noise_hat2, axis=1), axis=1)
        Noise_hat = self.Noise_hat2
        n_gmm = np.sum(self.p_ntl, 0)
        self.w_nl = np.expand_dims(n_gmm/np.expand_dims(np.sum(n_gmm, 0),0),1)
        return Signal_hat, Noise_hat
    def update_signal_noise_spp(self, x, spp):
        sap = 1 - spp
        e_bias = np.expand_dims(self.bias, 0)
        #alpha_d_hat = self.alpha_d + (1-self.alpha_d)*spp
        Signal_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_signal, 0)
        Signal_hat2 = x - (sap*x + spp*(np.sum(np.sum(Signal_hat2, axis=1), axis=1)))*self.alpha_d

        Noise_hat2 = np.expand_dims(self.p_o_t_k_l, -1)* np.expand_dims(self.mismatch_noise, 0)
        #Noise_hat2 = sap*x + spp*(x - np.sum(np.sum(Noise_hat2, axis=1), axis=1))
        Noise_hat2 = x - spp * np.sum(np.sum(Noise_hat2, axis=1), axis=1)

        n_gmm = np.sum(self.p_ntl, 0)
        self.w_nl = np.expand_dims(n_gmm/np.expand_dims(np.sum(n_gmm, 0),0),1)
        return Signal_hat2, Noise_hat2
if __name__ == '__main__':

    #means, stds, probs = phoneme_extract_gaussians()
    means, stds, probs = phoneme_extract_gaussians()
    home_dir = '/home/devpath/datasets/hybridwav/npy'
    w_f = os.path.join(home_dir, 'gmm04_001_00-SNR0DB.wav')
    #w_f = os.path.join(home_dir, 'gmm04_001_00.wav')
    wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
    meg1, angle1 = stft_analysis(wav_r, samplerate=16000,
                     winlen=0.025,
                     winstep=0.01
                     )
    meg1 = np.where(meg1 == 0, np.finfo(float).eps, meg1)
    logmeg1 = np.log(meg1)
    #feat = logfbank(wav_r, nfilt=26, samplerate=16000, winlen=0.025, winstep=0.010, winfunc=np.hamming)
    #$maptoheat3(feat, 'jet')
    #maptoheat3(meg1, 'jet')
    #maptoheat3(logmeg1, 'jet')
    #channel_image = np.transpose(feat, (1, 0))
    #maptoheat3(channel_image, 'jet')

    meg_pwr = np.square(meg1)
    low_fr = np.ones(17, float) * 2
    mid_fr = np.ones(48, float) * 2.8
    hi_fr = np.ones(192, float) * 5
    delta = np.concatenate((low_fr, mid_fr, hi_fr))

    imcra_noise = imcra(alpha_d=0.89, alpha_s=0.8, alpha_p=0.2, lambda_d=meg_pwr[0], frame_L=100,
                        fft_len=512,
                        delta=delta, beta=1.23, b_min=1.66, gamma0=4.6, gamma1=3, zeta0=1.67)
    m_present = np.zeros(257, float)
    for i, pwr in enumerate(meg_pwr):
        c_noise, G, P = imcra_noise.tracking_noise(pwr, i)

        m_present = np.vstack((m_present, P))
    m_present = m_present[1:]

    '''
    mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)*2
    stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)*2
    sigma = np.square(stand)
    g_m_max = gmm_phoneme_noise(np.array(means)*2, np.square(np.array(stds)*2), np.array(probs), mean, sigma, 3)
    '''
    mean = np.expand_dims(np.mean(logmeg1[:25, :], axis=0), 0)
    stand = np.expand_dims(np.std(logmeg1[:25, :], axis=0), 0)
    sigma = np.square(stand)
    g_m_max = gmm_phoneme_noise(np.array(means), np.square(np.array(stds)), np.array(probs), mean, sigma, 2)

    for ix in range(10):
        print('############################ process loop: ' + str(ix))
        g_m_max.compensate_model()
        g_m_max.calculate_P_o_t_k_l(logmeg1)
        #meg, noi = g_m_max.update_signal_noise(logmeg1)
        meg, noi = g_m_max.update_signal_noise_spp(logmeg1,m_present)
        #maptoheat3(meg, 'jet')
        #mag = np.exp(meg/2)
        mag = np.exp(meg)
        wav2 = polar_synthesis_np(mag, angle1, 400,160)
        wav2 = np.squeeze(wav2)
        path2 = os.path.join(home_dir, 'gmm_gen/gmm04_001_00-SNR0DB_'+str(ix)+'.wav')
        sf.write(path2, wav2, 16000)
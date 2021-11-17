import tensorflow as tf
import numpy as np
from tensorflow.python.ops.signal import window_ops
from scipy import stats
import decimal, math
import os, sys
import librosa
import soundfile as sf
import functools
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.special import exp1
import math

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


def mmse_lsa_np(xi, gamma):
    """
    Computes the MMSE-LSA gain function.

    Numpy version:
        v_1 = np.divide(xi, np.add(1.0, xi))
        nu = np.multiply(v_1, gamma)
        return np.multiply(v_1, np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.
    Argument/s:
        xi - a priori SNR.
        gamma - a posteriori SNR.

    Returns:
        MMSE-LSA gain function.
    """
    xi = np.where(xi == 0, np.finfo(float).eps, xi)
    gamma = np.where(gamma == 0, np.finfo(float).eps, gamma)
    v_1 = np.divide(xi, np.add(1.0, xi))
    nu = np.multiply(v_1, gamma)

    return np.multiply(v_1, np.exp(np.multiply(0.5, exp1(nu)))) # MMSE-LSA gain function.

class mcra(object):
    def __init__(self, alpha_d, alpha_s, alpha_p, lambda_d, frame_L, bin_num, delta, *tupleArg):

        self.alpha_d = np.expand_dims(alpha_d,0)
        self.alpha_s = np.expand_dims(alpha_s,0)
        self.alpha_p = np.expand_dims(alpha_p,0)
        if len(lambda_d.shape) == 2:
            self.lambda_d = lambda_d
        elif len(lambda_d.shape) == 1:
            self.lambda_d = np.expand_dims(lambda_d,0)
        self.bin_len = bin_num
        a = np.hanning(7)
        self.matrix = np.eye(self.bin_len)*a[3] \
                      + np.eye(self.bin_len, k=-2)*a[1] + np.eye(self.bin_len, k=2)*a[5] \
                      + np.eye(self.bin_len, k=-1)*a[2] + np.eye(self.bin_len, k=1)*a[4]

        self.matrix = np.expand_dims(self.matrix, 0).repeat(self.lambda_d.shape[0], 0)
        self.S = self.S_tmp = self.S_min = np.squeeze(np.matmul(self.matrix, np.expand_dims(self.lambda_d, -1)),-1)

        self.frame_L = frame_L
        self.delta = np.expand_dims(delta,0)
        self.self_alpha_D_hat = np.expand_dims(alpha_d,0)
        self.speech_present = np.expand_dims(np.zeros(self.bin_len, float),0)
        self.snr_gammar = np.expand_dims(np.ones(self.bin_len, float)*0.1,0)
        self.snr_xi = np.expand_dims(np.ones(self.bin_len, float)*0.1,0)
        self.alpha_snr = 0.92
        self.G_h=mmse_lsa_np(self.snr_xi, self.snr_gammar)
        self.G_min = np.expand_dims(np.ones(self.bin_len, float) * 0.09,0)

    def update_snr_dd(self, pwr):
        snr_gammar_prev = self.snr_gammar
        self.snr_gammar = pwr / self.lambda_d
        self.snr_xi = self.alpha_snr * np.square(self.G_h) * snr_gammar_prev + (1 - self.alpha_snr) * np.maximum(
            self.snr_gammar - 1, 0)

    def update_S(self, pwr):

        S_f = np.squeeze(np.matmul(self.matrix, np.expand_dims(pwr,-1)),-1)
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * S_f

    def tracking_S_win(self, current_frame):
        if current_frame % self.frame_L == 0:
            self.S_min = np.minimum(self.S, self.S_tmp)
            self.S_tmp = self.S
        else:
            self.S_min = np.minimum(self.S, self.S_min)
            self.S_tmp = np.minimum(self.S, self.S_tmp)


    def update_speech_present(self):

        S_ratio = self.S/self.S_min
        p = np.array(S_ratio > self.delta).astype(int)
        self.speech_present = self.alpha_p * self.speech_present + (1 - self.alpha_p) * p

    def update_alpha_d(self):
        self.alpha_D_hat = self.alpha_d + (1-self.alpha_d)*self.speech_present

    def update_noise(self, pwr):
        self.lambda_d = self.alpha_D_hat * self.lambda_d + (1 - self.alpha_D_hat) * pwr
    def update_SNR_GH(self):
        self.G_h = mmse_lsa_np(self.snr_xi, self.snr_gammar)

    def tracking_noise(self, pwr, c_frame):
        self.update_snr_dd(pwr)
        self.update_S(pwr)
        self.tracking_S_win(c_frame)
        self.update_speech_present()
        self.update_alpha_d()
        self.update_noise(pwr)
        self.update_SNR_GH()
        return np.squeeze(self.lambda_d), np.squeeze(self.G_h), np.squeeze(self.speech_present)

    def mmse_lsa(self, meg, c_frame):
        pwr = np.square(meg)
        lambda_d, G, P = self.tracking_noise(pwr, c_frame)
        return np.squeeze(G * meg)

    def omlsa(self, meg, c_frame):
        pwr = np.square(meg)
        lambda_d, G, P = self.tracking_noise(pwr, c_frame)
        return np.squeeze(np.power(G, P) * np.power(self.G_min, (1 - P)) * meg)

class mcra_2(mcra):
    def __init__(self, alpha_d, alpha_s, alpha_p, lambda_d, frame_L, fft_len, delta, gamma, beta):
        super().__init__(alpha_d, alpha_s, alpha_p, lambda_d, frame_L, fft_len, delta)
        self.gamma = gamma
        self.beta = beta
        self.S_minus_one = self.S

    def update_S_2(self,meg):
        self.S_minus_one = self.S
        self.update_S(meg)

    def tracking_S_continue(self):
        p = np.array(self.S_min < self.S).astype(int)
        p_not = np.array(self.S_min >= self.S).astype(int)
        self.S_min = self.S*p_not+(self.gamma * self.S_min + (1-self.gamma)*(self.S - self.beta * self.S_minus_one)/(1-self.beta))*p

    def tracking_noise(self, pwr, c_frame):
        self.update_snr_dd(pwr)
        self.update_S_2(pwr)
        self.tracking_S_continue()
        self.update_speech_present()
        self.update_alpha_d()
        self.update_noise(pwr)
        self.update_SNR_GH()
        return self.lambda_d, self.G_h, self.speech_present



class imcra(mcra):
    def __init__(self, alpha_d, alpha_s, alpha_p, lambda_d, frame_L, fft_len, delta, beta, b_min, gamma0, gamma1, zeta0):

        super().__init__(alpha_d, alpha_s, alpha_p, lambda_d, frame_L, fft_len, delta)
        self.beta = beta
        self.b_min = b_min
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.zeta0 = zeta0
        self.S_hat = self.S
        self.S_min_hat = self.S_min
        self.S_tmp_hat = self.S_tmp
        self.zero = np.zeros(self.bin_len, float)
        self.ones = np.ones(self.bin_len, float)
        self.gamma1minus1 = self.gamma1 - self.ones

        self.alpha_s_hat = self.alpha_s * 1.2
        self.frame_L_hat = frame_L * 0.5


    def update_S_hat(self, pwr):
        gamma_min = pwr/(self.b_min*self.S_min)
        zeta = self.S/(self.b_min*self.S_min)
        I_tmp = np.array(np.logical_and((gamma_min < self.gamma0), (zeta < self.zeta0))).astype(int)
        win_I = np.matmul(self.matrix, I_tmp)
        a_p = np.array(win_I == self.zero).astype(int)
        a_p_not = np.array(win_I > self.zero).astype(int)
        denominator = win_I + a_p
        numerator = win_I*pwr + self.S_hat*a_p#_not
        S_f = numerator/denominator
        self.S_hat = self.alpha_s_hat * self.S_hat + (1-self.alpha_s_hat)*S_f

    def tracking_S_win_hat(self, current_frame):
        if current_frame % self.frame_L_hat == 0:
            self.S_min_hat = np.minimum(self.S_hat, self.S_tmp_hat)
            self.S_tmp_hat = self.S_hat
        else:
            self.S_min_hat = np.minimum(self.S_hat, self.S_min_hat)
            self.S_tmp_hat = np.minimum(self.S_hat, self.S_tmp_hat)

    def update_speech_present(self,pwr):
        gamma_min_hat = pwr/(self.b_min*self.S_min_hat)
        zeta_hat = self.S_hat/(self.b_min*self.S_min_hat)
        a = np.array(np.logical_and((gamma_min_hat < self.ones),(zeta_hat < self.zeta0))).astype(int)
        b = np.array(np.logical_and((zeta_hat < self.zeta0), np.logical_and((gamma_min_hat < self.gamma1), (gamma_min_hat > self.ones)))).astype(int)
        q = a + b*(self.gamma1-gamma_min_hat)/self.gamma1minus1
        c_x = 1+self.snr_xi
        c_x = np.where(c_x == 0, np.finfo(float).eps, c_x)
        v = np.true_divide(self.snr_xi*self.snr_gammar,c_x)
        oneminusq = 1-q
        oneminusq = np.where(oneminusq == 0, np.finfo(float).eps, oneminusq)
        sp_reciprocal = 1+q*(1+self.snr_xi)*np.exp(-v)/oneminusq
        sp_reciprocal = np.where(sp_reciprocal == 0, np.finfo(float).eps, sp_reciprocal)
        self.speech_present = 1/sp_reciprocal

    def tracking_noise(self, pwr, c_frame):
        self.update_snr_dd(pwr)
        self.update_S(pwr)
        self.tracking_S_win(c_frame)
        self.update_S_hat(pwr)
        self.tracking_S_win_hat(c_frame)
        self.update_speech_present(pwr)
        self.update_alpha_d()
        self.update_noise(pwr)
        self.update_SNR_GH()
        return np.squeeze(self.lambda_d), np.squeeze(self.G_h), np.squeeze(self.speech_present)
''''''
class mcra_tbrr(mcra):

    def __init__(self, alpha_d, alpha_s, alpha_p, lambda_d, z_b, z_r, frame_L, bin_num, delta, *tupleArg):
        super().__init__(alpha_d, alpha_s, alpha_p, lambda_d, frame_L, bin_num, delta)
        self.mcra_zb = mcra(alpha_d=alpha_d, alpha_s=alpha_s, alpha_p=alpha_p, lambda_d=z_b, frame_L=frame_L, bin_num=bin_num,
                              delta=delta)
        self.mcra_zr = mcra(alpha_d=alpha_d, alpha_s=alpha_s, alpha_p=alpha_p, lambda_d=z_r, frame_L=frame_L, bin_num=bin_num,
                            delta=delta)
        self.Lambda_0 = 1.67
        self.Lambda_1 = 1.81
        self.gammar_0 = 4.6
        self.gammar_0_minus_1 = 4.6-1
        self.Omega_low = 1
        self.Omega_high = 3
        self.Omega_delta =self.Omega_high - self.Omega_low
        self.betta = 1.47

    def tracking_tbrr(self, pwr_b, pwr_bm, c_frame):
        self.Q_zb,self.G_zb, _ = self.mcra_zb.tracking_noise(pwr_b, c_frame)
        self.Q_zr,self.G_zr, _ = self.mcra_zr.tracking_noise(pwr_bm, c_frame)
        self.Lambda_y = np.squeeze(self.mcra_zb.S/self.mcra_zb.lambda_d)
        self.Lambda_bm = np.max(self.mcra_zr.S / self.mcra_zr.lambda_d,axis=0)
        self.Omega = (self.mcra_zb.S - self.mcra_zb.lambda_d)/np.max(self.mcra_zr.S - self.mcra_zr.lambda_d,axis=0)
        H0 = np.array(self.Lambda_y <= self.Lambda_0).astype(int)
        H0_not_mask = 1 - H0
        H1_tmp = np.array(self.Lambda_bm <= self.Lambda_1).astype(int)
        H1 = H0_not_mask* H1_tmp
        H1_not_mask = 1 - H1
        Hr = H0_not_mask * H1_not_mask
        H0t = np.logical_or(np.array(self.Omega < self.Omega_low), np.array(self.snr_gammar < 1)).astype(int)
        H0t_tbrr = H0t * Hr
        H0t_tbrr_not_mask = 1 - H0t_tbrr
        H_tbrr_mask = Hr * H0t_tbrr_not_mask
        H1_tbrr = np.logical_or(np.array(self.Omega > self.Omega_high), np.array(self.snr_gammar > self.gammar_0)).astype(int)
        H1_tbrr_r = H1_tbrr * H_tbrr_mask
        H1_tbrr_r_not_mask = 1 - H1_tbrr_r
        Hr_tbrr_mask = H_tbrr_mask * H1_tbrr_r_not_mask
        r_tbrr = np.maximum((self.gammar_0 - self.snr_gammar)/self.gammar_0_minus_1, (self.Omega_high-self.Omega)/self.Omega_delta)
        Hr_tbrr = r_tbrr * Hr_tbrr_mask
        self.q_tbrr = H0+H0t_tbrr+Hr_tbrr

    def update_speech_present(self):
        c_x = 1 + self.snr_xi
        c_x = np.where(c_x == 0, np.finfo(float).eps, c_x)
        v = np.true_divide(self.snr_xi * self.snr_gammar, c_x)
        oneminusq = 1 - self.q_tbrr
        oneminusq = np.where(oneminusq == 0, np.finfo(float).eps, oneminusq)
        sp_reciprocal = 1 + self.q_tbrr * (1 + self.snr_xi) * np.exp(-v) / oneminusq
        sp_reciprocal = np.where(sp_reciprocal == 0, np.finfo(float).eps, sp_reciprocal)
        self.speech_present = 1 / sp_reciprocal

    def tracking_noise(self, pwr, pwr_b, pwr_bm, c_frame):
        self.update_snr_dd(pwr)
        #self.update_S(pwr)
        #self.tracking_S_win(c_frame)
        self.tracking_tbrr(pwr_b, pwr_bm, c_frame)
        self.update_speech_present()
        self.update_alpha_d()
        self.update_noise(pwr)
        self.update_SNR_GH()
        return self.lambda_d, self.G_h, self.speech_present

    def omlsa(self, meg, meg_b, meg_bm, c_frame):
        pwr = np.square(meg)
        pwr_b = np.square(meg_b)
        pwr_bm = np.square(meg_bm)
        lambda_d, G, P = self.tracking_noise(pwr, pwr_b, pwr_bm,c_frame)
        return np.squeeze(np.power(G, P) * np.power(self.G_min, (1 - P)) * meg)


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)

def expw(x):
    c_exp_x0 = [
        [1.0, 1.0644944589178593, 1.1331484530668263, 1.2062302494209807,
         1.2840254166877414, 1.3668379411737963, 1.4549914146182013, 1.5488302986341331,
         1.6487212707001282, 1.7550546569602985, 1.8682459574322223, 1.988737469582292,
         2.117000016612675, 2.2535347872132085, 2.398875293967098, 2.553589458062927],
        [1.0, 1.0039138893383475, 1.007843097206448, 1.0117876835593316,
         1.0157477085866857, 1.0197232327137742, 1.023714316602358, 1.0277210211516217,
         1.0317434074991028, 1.035781537021624, 1.03983547133623, 1.0439052723011284,
         1.0479910020166328, 1.0520927228261099, 1.056210497316932, 1.0603443883214314],
        [1.0, 1.0002441704297478, 1.0004884004786945, 1.000732690161397,
         1.0009770394924165, 1.0012214484863171, 1.0014659171576668, 1.001710445521037,
         1.0019550335910028, 1.0021996813821428, 1.002444388909039, 1.0026891561862772,
         1.0029339832284467, 1.0031788700501403, 1.0034238166659546, 1.003668823090489]
    ]
    if x<-709:  return 0
    elif x>709: return 1.7E+308
    s = x * np.log2(np.e)
    integer = np.floor(s)
    decimal = (s - np.floor(s))*np.log(2)

    ep = decimal * 16
    q0 = int(np.floor(ep))
    ep = ep - np.floor(ep)
    ep1 = ep * 16
    q1 = int(np.floor(ep1))
    ep1 = ep1 - np.floor(ep1)
    ep2 = ep1 * 16
    q2 = int(np.floor(ep2))
    ep2 = ep2 - np.floor(ep2)

    h = c_exp_x0[0][q0] * c_exp_x0[1][q1] * c_exp_x0[2][q2]
    h1 = np.exp(q0/16)*np.exp(q1/(16*16))*np.exp(q2/(16*16*16))
    w = ep2 / 4096
    ew = 1 + w + w * w / 2 + w * w * w / 6 + w * w * w * w / 24
    eww = np.exp(w)
    decimal_final = h * ew
    result = decimal_final * 2**integer

    golden = np.exp(x)
    goldenn = 2**(np.log2(np.e)*x)
    pass


def loge(x):
    #if x <= 0:  return -1.7E+308
    #elif x > 100000000: return 18.420680743952367
    decimal = 0
    shift = 1
    inverselist = np.flipud(np.arange(52))

    for i in inverselist:
        mask = 1 << i
        shift /= 2
        if mask & x:
            decimal += shift


if __name__ == '__main__':
    pi = math.pi
    M = 256
    b = np.exp(1j)
    W = np.exp((2*pi/M)*1j)
    nature = np.arange(M)

    #expw(3.5)
    loge(0x5000000000000)
    DFT_Matrix = np.ones([M,M],np.complex)
    for row in range(M):
        DFT_Matrix[row]=W**(-nature*(row))
    def exp11(x):
        return np.exp(-x)/x



    x = np.linspace(0, 8, 256)
    y = exp11(x)
    plt.plot(x, y)
    plt.show()
    """
         三角信号
    

    def triangle_wave(x, c, hc):  # 幅度为hc，宽度为c,斜度为hc/2c的三角波
        if x >= c / 2:
            r = 0.0
        elif x <= -c / 2:
            r = 0.0
        elif x > -c / 2 and x < 0:
            r = 2 * x / c * hc + hc
        else:
            r = -2 * x / c * hc + hc
        return r


    x = np.linspace(-3, 3, 256)
    y = np.array([triangle_wave(t, 4.0, 1.0) for t in x])
    plt.ylim(-0.2, 1.2)
    plt.plot(x, y)
    plt.show()

    #Y = DFT_Matrix*y
    Y = np.matmul(DFT_Matrix, y)
    y_idx = np.linspace(0, 2*pi, 256)
    plt.plot(y_idx, np.absolute(Y))
    """
    """
         矩形脉冲信号
    

    def rect_wave(x, c, c0):  # 起点为c0，宽度为c的矩形波
        if x >= (c + c0):
            r = 0.0
        elif x < c0:
            r = 0.0
        else:
            r = 1
        return r


    x = np.linspace(-2, 4, 256)
    y = np.array([rect_wave(t, 2.0, -1.0) for t in x])
    plt.ylim(-0.2, 4.2)
    plt.plot(x, y)
    plt.show()

    Y = np.matmul(DFT_Matrix, y)
    y_idx = np.linspace(0, 2*pi, 256)
    plt.plot(y_idx, np.absolute(Y))
    """
    from sympy import plot, sin, Symbol
    x = np.linspace(0, 8, 256)
    y = np.array([sin(np.pi/4*t) for t in x])
    plt.ylim(-1.2, 6.2)
    plt.plot(x, y)
    plt.show()
    y=y.astype(np.float64)
    Y = np.matmul(DFT_Matrix, y)
    y_idx = np.linspace(0, 2 * pi, 256)
    plt.plot(y_idx, np.absolute(Y))


    dd = 128
    nature1 = np.arange(dd)
    H_0 = np.exp(-(2 * pi * nature1 / dd) * 1j)
    W = np.exp((2 * pi / dd) * 1j)
    ranges = range(1, dd)
    H_w = np.zeros(dd,np.complex)
    for omega in nature1:
        #tm = W**(-(nature1*omega))
        #tm = np.exp(-(2 * pi * nature1 / dd + omega) * 1j)
        tm = np.exp(-(2 * pi * nature1 * omega / dd) * 1j)
        H_w[omega] = np.sum(tm)
    abs_H = np.abs(H_w)
    plt.figure(figsize=(20, 10))

    plt.plot(ranges, abs_H[1:], 'b--o', label='H(jw)')
    plt.show()


    print("Processing observations...")
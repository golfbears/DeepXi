import tensorflow as tf
import numpy as np
from hybrid.analyze_label import ensures_dir

import os, sys
import librosa
import soundfile as sf
from mcra.mcra123 import tensor_polar
import matplotlib.pyplot as plt

class histogram():
    def __init__(self, x, alpha_d=0.8, alpha_s=0.9, frame_L=40, fft_len=512, delta=8):
        self.alpha_d = alpha_d
        self.alpha_s = alpha_s
        self.bin_len = int(np.floor(fft_len / 2) + 1)
        self.S = x
        self.frame_L = frame_L
        self.S_pool = np.zeros([frame_L, self.bin_len])
        self.delta = delta
        self.delta_snr = 2.5
        self.std_gain = 1.99
    def update_S(self, pwr, current_frame):
        pool_idx = int(current_frame % self.frame_L)
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * pwr
        self.S_pool[pool_idx] = self.S

    def tracking_histogram_win(self, current_frame):
        if current_frame == 0:
            self.noise = self.S_pool[0]
            self.mean = self.noise
            self.std = np.zeros(self.bin_len, float)
            self.std = np.where(self.std == 0, np.finfo(float).eps, self.std)
        elif current_frame < self.frame_L :
            self.noise = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean = self.noise
            self.std = np.std(self.S_pool[:current_frame], axis=0)
            self.std = np.where(self.std == 0, np.finfo(float).eps, self.std)
        else:
            for idx in range(257):
                hists,bins = np.histogram(self.S_pool[:,idx], 40)
                a = np.argmax(hists)
                h_max = bins[a]
                self.noise[idx] = self.alpha_d * self.noise[idx] + (1-self.alpha_d) * h_max
                self.mean[idx] = self.alpha_d * self.mean[idx] + (1-self.alpha_d) * np.mean(self.S_pool[:,idx])
                self.std[idx] = self.alpha_d * self.std[idx] + (1-self.alpha_d) * np.std(self.S_pool[:,idx])


    def tracking_hist_snr(self, current_frame):
        if current_frame == 0:
            self.noise1 = self.S_pool[0]
            self.mean1 = self.noise1
            self.std1 = np.zeros(self.bin_len, float)
            self.std1 = np.where(self.std1 == 0, np.finfo(float).eps, self.std1)
        elif current_frame < self.frame_L:
            self.noise1 = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean1 = self.noise1
            self.std1 = np.std(self.S_pool[:current_frame], axis=0)
            self.std1 = np.where(self.std1 == 0, np.finfo(float).eps, self.std1)
        else:
            self.noise1 = np.where(self.noise1 == 0, np.finfo(float).eps, self.noise1)
            tmp_snr_hat = self.S_pool / np.expand_dims(self.noise1, axis=0)
            #delta_threshold = self.noise1 + self.delta
            #mean_threshold = np.mean(self.S_pool, axis=0)
            #threshold = np.minimum(delta_threshold,mean_threshold)
            snr_mask = tmp_snr_hat < self.delta_snr
            #thr_mask = self.S_pool < threshold
            for idx in range(257):
                true_aray = snr_mask[:,idx]
                arg_idx = np.squeeze(np.argwhere(true_aray == False))
                if arg_idx.any():#len(arg_idx) == 0:
                    aray = np.delete(self.S_pool[:, idx], arg_idx)
                else:
                    aray = self.S_pool[:, idx]
                pure_len = len(aray)
                if pure_len > 5:
                    hists,bins = np.histogram(aray, np.minimum(pure_len,40))
                    a = np.argmax(hists)
                    h_max = bins[a]
                    self.noise1[idx] = self.alpha_d * self.noise1[idx] + (1-self.alpha_d) * h_max
                if pure_len > 2:
                    self.mean1[idx] = self.alpha_d * self.mean1[idx] + (1 - self.alpha_d) * np.mean(aray)
                    self.std1[idx] = self.alpha_d * self.std1[idx] + (1 - self.alpha_d) * np.std(aray)


    def tracking_hist_threshold(self, current_frame):
        if current_frame == 0:
            self.noise2 = self.S_pool[0]
            self.mean2 = self.noise2
            self.std2 = np.zeros(self.bin_len, float)
            self.std2 = np.where(self.std2 == 0, np.finfo(float).eps, self.std2)
        elif current_frame < self.frame_L:
            self.noise2 = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean2 = self.noise2
            self.std2 = np.std(self.S_pool[:current_frame], axis=0)
            self.std2 = np.where(self.std2 == 0, np.finfo(float).eps, self.std2)
        else:
            #tmp_snr_hat = self.S_pool / np.expand_dims(self.noise2, axis=0)
            delta_threshold = self.noise1 + self.delta
            mean_threshold = np.mean(self.S_pool, axis=0)
            threshold = np.minimum(delta_threshold,mean_threshold)
            #snr_mask = tmp_snr_hat < self.delta_snr
            thr_mask = self.S_pool < np.expand_dims(threshold, axis=0)
            for idx in range(257):
                true_aray = thr_mask[:, idx]
                arg_idx = np.squeeze(np.argwhere(true_aray == False))
                if arg_idx.any():  # len(arg_idx) == 0:
                    aray = np.delete(self.S_pool[:, idx], arg_idx)
                else:
                    aray = self.S_pool[:, idx]
                pure_len = len(aray)
                if pure_len > 5:
                    hists, bins = np.histogram(aray, np.minimum(pure_len,40))
                    a = np.argmax(hists)
                    h_max = bins[a]
                    self.noise2[idx] = self.alpha_d * self.noise2[idx] + (1 - self.alpha_d) * h_max
                if pure_len > 2:
                    self.mean2[idx] = self.alpha_d * self.mean2[idx] + (1 - self.alpha_d) * np.mean(aray)
                    self.std2[idx] = self.alpha_d * self.std2[idx] + (1 - self.alpha_d) * np.std(aray)


    def tracking_noise(self, pwr, c_frame):
        self.update_S(pwr, c_frame)
        self.tracking_histogram_win(c_frame)
        self.tracking_hist_snr(c_frame)
        self.tracking_hist_threshold(c_frame)
        return self.noise, self.noise1, self.noise2
    def tracking_mu_std(self, pwr, c_frame):
        self.update_S(pwr, c_frame)
        self.tracking_histogram_win(c_frame)
        self.tracking_hist_snr(c_frame)
        self.tracking_hist_threshold(c_frame)
        return self.mean, self.std1, self.mean1, self.std1, self.mean2, self.std2
    def get_mu_std(self):
        return self.mean, self.std1, self.mean1, self.std1, self.mean2, self.std2

if __name__ == '__main__':

    clean_dir = '/home/devpath/datasets/tmp/clean'
    #home_dir = '/home/devpath/datasets/new/small'  # '/home/devpath/golfbears/DeepXi/set/test_noisy_speech'
    home_dirs = [
        '/home/devpath/datasets/tmp/noisy'
        # '/home/HDD/jzli/out0528_4mic_wpe/clean',
        # '/home/HDD/jzli/out0528_4mic_wpe/crowd',
        # '/home/HDD/jzli/out0528_4mic_wpe/music',
        # '/home/HDD/jzli/out0528_4mic_wpe/fans'
    ]

    analysis_path_mcra = os.path.join(clean_dir,'mcra')
    ensures_dir(analysis_path_mcra)
    analysis_path_mcra_2 = os.path.join(clean_dir,'mcra_2')
    ensures_dir(analysis_path_mcra_2)
    analysis_path_imcra = os.path.join(clean_dir,'imcra')
    ensures_dir(analysis_path_imcra)
    low_fr = np.ones(17, float) * 2
    mid_fr = np.ones(48, float) * 2.8
    hi_fr = np.ones(192, float) * 5
    delta = np.concatenate((low_fr, mid_fr, hi_fr))
    G_min = np.ones(257, float) * 0.09
    t_polar = tensor_polar(400,160,512,16000)

    for home_dir in home_dirs:
        w_list = os.listdir(home_dir)

        for w in w_list:
            if not w.__contains__('.wav'):
                continue
            analysis_path = os.path.join(clean_dir,w.split('.')[0]+'_plot')
            ensures_dir(analysis_path)
            w_f = os.path.join(home_dir, w)
            wav_r, f_s = librosa.load(w_f, sr=16000, mono=True, dtype=np.float32)
            meg1, angle1 = t_polar.polar_analysis(wav_r)
            mcra_noise = histogram(delta, alpha_d=0.95, alpha_s=0.8, frame_L=100, fft_len=512,
                              delta=5)
            mcra_2_noise = histogram(delta, alpha_d=0.89, alpha_s=0.8, frame_L=100, fft_len=512,
                                  delta=delta)

            imcra_noise = histogram(delta, alpha_d=0.89, alpha_s=0.8, frame_L=100, fft_len=512,
                                delta=delta)
            m_noise = np.zeros(257, float)
            m_noise_2 = np.zeros(257, float)
            m_noise_i = np.zeros(257, float)
            m_G = np.zeros(257, float)
            m_G_2 = np.zeros(257, float)
            m_G_i = np.zeros(257, float)
            m_omlsa = np.zeros(257, float)
            m_omlsa_2 = np.zeros(257, float)
            m_omlsa_i = np.zeros(257, float)
            m_present = np.zeros(257, float)
            m_present_2 = np.zeros(257, float)
            m_present_i = np.zeros(257, float)

            for i, meg in enumerate(meg1):
                pwr = np.square(meg)
                c_noise, G, P = mcra_noise.tracking_noise(pwr, i)
                m_noise = np.vstack((m_noise, c_noise))
                m_G = np.vstack((m_G, G))
                m_present = np.vstack((m_present, P))
                omlsa = np.power(G, P) * np.power(G_min, (1 - P)) * meg
                m_omlsa = np.vstack((m_omlsa, omlsa))
                c_noise_2, G, P = mcra_2_noise.tracking_noise(pwr, i)
                m_noise_2 = np.vstack((m_noise_2, c_noise_2))
                m_G_2 = np.vstack((m_G_2, G))
                m_present_2 = np.vstack((m_present_2, P))
                omlsa = np.power(G, P) * np.power(G_min, (1 - P)) * meg
                m_omlsa_2 = np.vstack((m_omlsa_2, omlsa))
                c_noise_i, G, P = imcra_noise.tracking_noise(pwr, i)
                m_noise_i = np.vstack((m_noise_i, c_noise_i))
                m_G_i = np.vstack((m_G_i, G))
                m_present_i = np.vstack((m_present_i, P))
                omlsa = np.power(G, P) * np.power(G_min, (1 - P)) * meg
                m_omlsa_i = np.vstack((m_omlsa_i, omlsa))

            m_noise = m_noise[1:]
            m_noise_2 = m_noise_2[1:]
            m_noise_i = m_noise_i[1:]
            m_G = m_G[1:]
            m_G_2 = m_G_2[1:]
            m_G_i = m_G_i[1:]
            m_omlsa = m_omlsa[1:]
            m_omlsa_2 = m_omlsa_2[1:]
            m_omlsa_i = m_omlsa_i[1:]
            m_present = m_present[1:]
            m_present_2 = m_present_2[1:]
            m_present_i = m_present_i[1:]

            wav = t_polar.polar_synthesis(m_omlsa, angle1)
            path2 = os.path.join(analysis_path_mcra, w.split('.')[0] + '.wav')
            sf.write(path2, wav.numpy(), 16000)
            wav = t_polar.polar_synthesis(m_omlsa_2, angle1)
            path2 = os.path.join(analysis_path_mcra_2, w.split('.')[0] + '.wav')
            sf.write(path2, wav.numpy(), 16000)
            wav = t_polar.polar_synthesis(m_omlsa_i, angle1)
            path2 = os.path.join(analysis_path_imcra, w.split('.')[0] + '.wav')
            sf.write(path2, wav.numpy(), 16000)

            for bin in range(257):
                plt.figure(figsize=(20, 10))
                plt.suptitle('Tracking noise at bin: ' + str(bin))

                plt.subplot(2, 2, 1)
                epochs = range(1, np.shape(m_noise)[0] + 1)
                plt.plot(epochs, m_noise[:, bin], 'b--o', label='mcra_noise')
                plt.plot(epochs, m_noise_i[:, bin], 'c--o', label='imcra_noise')
                plt.plot(epochs, m_noise_2[:, bin], 'g--o', label='mcra_2_noise')
                plt.title('tracking noise')
                plt.xlabel('frame Numbers')
                plt.ylabel('mu')
                plt.legend()
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.plot(epochs, m_G[:, bin], 'b--o', label='mcra_G')
                plt.plot(epochs, m_G_i[:, bin], 'c--o', label='imcra_G')
                plt.plot(epochs, m_G_2[:, bin], 'g--o', label='mcra_2_G')
                plt.title('tracking G')
                plt.xlabel('frame numbers')
                plt.ylabel('sigma')
                plt.legend()
                plt.grid()

                plt.subplot(2, 2, 2)
                plt.plot(epochs, m_present[:, bin], 'b--o', label='mcra_present')
                plt.plot(epochs, m_present_i[:, bin], 'c--o', label='imcra_present')
                plt.plot(epochs, m_present_2[:, bin], 'g--o', label='mcra_2_present')
                plt.title('tracking speech present')
                plt.xlabel('frame numbers')
                plt.ylabel('sigma')
                plt.legend()
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.plot(epochs, meg1[:, bin], 'r--o', label='noisy_signal')
                plt.plot(epochs, m_omlsa[:, bin], 'b--o', label='mcra_omlsa')
                plt.plot(epochs, m_omlsa_i[:, bin], 'c--o', label='imcra_omlsa')
                plt.plot(epochs, m_omlsa_2[:, bin], 'g--o', label='mcra_2_omlsa')
                plt.title('tracking magnitude')
                plt.xlabel('frame numbers')
                plt.ylabel('sigma')
                plt.legend()
                plt.grid()

                fig_title = os.path.join(analysis_path, 'Tracking_noise_' + str(bin) + '_bin.png')
                plt.savefig(fig_title)

                # plt.show()
                plt.close()


        print("Processing observations...")
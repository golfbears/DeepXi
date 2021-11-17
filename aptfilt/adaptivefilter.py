import tensorflow as tf
import numpy as np
from hybrid.analyze_label import ensures_dir

import os, sys
import librosa
import soundfile as sf
from mcra.mcra123 import tensor_polar
import matplotlib.pyplot as plt

class lms_aptfilt():
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
        elif current_frame < self.frame_L :
            self.noise = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean = self.noise
            self.std = np.std(self.S_pool[:current_frame], axis=0)
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
        elif current_frame < self.frame_L:
            self.noise1 = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean1 = self.noise1
            self.std1 = np.std(self.S_pool[:current_frame], axis=0)
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
        elif current_frame < self.frame_L:
            self.noise2 = np.mean(self.S_pool[:current_frame], axis=0)
            self.mean2 = self.noise2
            self.std2 = np.std(self.S_pool[:current_frame], axis=0)
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
    import numpy as np
    import matplotlib.pylab as plt
    import padasip as pa
    import soundfile as sf

    # define path
    home_dir = '/home/devpath/ml/crowd'
    f_list = os.listdir(home_dir)
    for file in f_list:
        track, sr = sf.read(os.path.join(home_dir,file))
        '''
        # show results
        plt.figure(figsize=(15, 9))
        plt.subplot(211);
        plt.title("Adaptation");
        plt.xlabel("samples - k")
        plt.plot(track[:,0], 'b--o', label="d - target")
        plt.plot(track[:,1], 'g--', label="y - output");
        plt.legend()
        plt.subplot(212);
        plt.title("Filter error");
        plt.xlabel("samples - k")
        plt.plot(track[:,2], "r", label="e - error [dB]");
        plt.plot(track[:,3], "r", label="e - error [dB]");
        plt.legend()
        plt.tight_layout()
        plt.show()
        '''
        # creation of data
        N = 500
        x = np.random.normal(0, 1, (N, 4))  # input matrix
        v = np.random.normal(0, 0.1, N)  # noise
        d = 2 * x[:, 0] + 0.1 * x[:, 1] - 4 * x[:, 2] + 0.5 * x[:, 3] + v  # target

        # identification
        f = pa.filters.FilterLMS(n=32, mu=0.1, w="random")
        #y, e, w = f.run(d, x)

        x0 = track[:,3]
        d_t = track[:,1]
        '''
        x0 = track[:, 0]
        d_t = track[:, 1]
        '''
        x_matrix1 = np.stack([np.roll(x0,i) for i in range(32)])
        x_t = x_matrix1.transpose((1,0))
        y, e, w = f.run(d_t, x_t)

        # show results
        plt.figure(figsize=(15, 9))
        plt.subplot(211);
        plt.title("Adaptation");
        plt.xlabel("samples - k")
        plt.plot(track[:,1], 'b--', label="d - target")
        #plt.plot(track[:,3], 'g--', label="y - output");
        plt.legend()
        plt.subplot(212);
        plt.title("Filter error");
        plt.xlabel("samples - k")
        plt.plot(e, "r", label="e - error [dB]");
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("Processing observations...")
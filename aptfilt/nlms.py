import matplotlib.pylab as plt
import padasip as pa
import numpy as np
import os
from scipy.io import wavfile
from scipy.fftpack import fft
#from IPython.display import Audio
#import IPython
import soundfile as sf
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
    # constants
    # define path
    r_dir = '/home/devpath/wcs/noisetracking'
    home_dir = '/home/devpath/wcs/noisetracking/clean'
    analysis_path = os.path.join(r_dir, 'enhance')
    FILENAME = os.path.join(home_dir, 'clean_speech.wav')
    SAMPLERATE = 44100
    n = 300 # filter size
    D = 200 # signal delay


    # open and process source data
    fs, data = wavfile.read(FILENAME)
    y = data.copy()
    y = y.astype("float64")
    y = zs(y) / 10
    N = len(y)

    # contaminated with noise
    q = np.sin(2*np.pi*1000/99*np.arange(N) + 10.1 * np.sin(2*np.pi/110*np.arange(N)))
    d = y + q
    yp, e, w = nlms_block(d)
    # prepare data for simulation

    plt.figure(figsize=(15, 9))
    plt.subplot(311)
    plt.title("Adaptation")
    plt.xlabel("samples - k")
    plt.plot(q, "b", label="predict - target")
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
    plt.plot(d, "g", label="error between predict and input")
    # plt.plot(d_t, "b", label="d - input")
    plt.legend()
    plt.tight_layout()
    plt.show()
    path1 = os.path.join(analysis_path, 'nlms_predict' + '.wav')
    sf.write(path1, yp, 16000)
    path1 = os.path.join(analysis_path, 'nlms_error' + '.wav')
    sf.write(path1, e, 16000)
    path1 = os.path.join(analysis_path, 'nlms_noisy' + '.wav')
    sf.write(path1, d, 16000)
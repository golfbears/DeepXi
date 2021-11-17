import functools
import os
import collections
import contextlib
import sys
import wave
import tensorflow as tf
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



def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n



def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        sys.stdout.write(
            '1' if vad.is_speech(frame.bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp,))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def measure_x():
    # it produces input vector of size 3
    x = np.random.random(3)
    return x


def measure_d(x):
    # meausure system output
    d = 2 * x[0] + 1 * x[1] - 1.5 * x[2]
    return d

def measure_n(x):
    ln = len(x)
    if ln == 0:
        o = np.random.random(10)
    elif len(x) < 10:
        b = np.random.random(10-len(x))
        o = np.concatenate((x,b))
    else:
        o = x[-10:]
    return o

def main(args):



    if len(args) != 2:
        sys.stderr.write(
            'Usage: example.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    t_polar = tensor_polar(400, 160, 512, 16000)
    audio, sample_rate = read_wave(args[1])
    track, sr = sf.read(args[1])
    vad = webrtcvad.Vad(int(args[0]))
    frames = frame_generator(20, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    for i, segment in enumerate(segments):
        # path = 'chunk-%002d.wav' % (i,)
        print('--end')
        # write_wave(path, segment, sample_rate)
    x0 = track[:, 1]
    d_t = track[:, 3]

    filt = pa.filters.FilterLMS(10, mu=0.7)
    log_d = np.zeros(len(x0))
    for i, k in enumerate(x0):

        d_o = measure_n(d_t[:i + 1])
        int_d_o = measure_n(int_d_t[:i + 1])
        is_speech = vad.is_speech(int_d_o.tobytes(), sr)
        if not is_speech:
            filt.adapt(d_o, k)
        y = filt.predict(d_o)
        log_d[i] = y

        ### show results
    for i in range(10):
        log_d[i] = x0[i]
    plt.figure(figsize=(15, 9))
    plt.subplot(211);
    plt.title("Adaptation");
    plt.xlabel("samples - k")
    plt.plot(log_d, "b", label="d - target")
    # plt.plot(x0, "g", label="x - input")
    plt.legend()
    plt.subplot(212);
    plt.title("Filter error");
    plt.xlabel("samples - k")
    # plt.plot(10 * np.log10((log_d - x0) ** 2), "r", label="e - error [dB]")
    plt.plot(x0, "g", label="x - input")
    plt.plot(d_t, "b", label="d - input")
    plt.legend();
    plt.tight_layout();
    plt.show()





if __name__ == '__main__':
    main(sys.argv[1:])


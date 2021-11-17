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


if __name__ == '__main__':

    home_dir = '/home/devpath/datasets/speech_corpus/shellshell'
    source_dir = os.path.join(home_dir,'shell_train_noisy')
    csv_dir = os.path.join(home_dir,'aishell-train-initial-final-asr-lexicon.csv')
    tgt_csv = os.path.join(home_dir,'shell-train-noisy-initial-final-asr-lexicon.csv')

    src_list = os.listdir(source_dir)
    with open(tgt_csv, 'w') as tgt_csv:
        with open(csv_dir, 'r') as csv_dict:
            for i, a_line in enumerate(csv_dict):
                #print(a_line + ':' + str(i))
                new_lines = a_line.split(',')
                if not os.path.exists(os.path.join(home_dir,new_lines[0])):
                    tgt_csv.writelines(a_line)
                    continue

                key = new_lines[0].split('/')[-1]
                key = key.split('.')[0]
                for fi in src_list:
                    if fi.__contains__(key):
                        b_line = 'kws_train_noisy/'+fi
                        #src = os.path.join(home_dir,new_lines[0])
                        #src_raw,sr = librosa.load(src, sr=16000, mono=True, dtype=np.float32)
                        #src_len = len(src_raw)

                        #tgt = os.path.join(home_dir, b_line)
                        #tgt_raw, sr = librosa.load(tgt, sr=16000, mono=True, dtype=np.float32)
                        #tgt_len = len(src_raw)
                        b_line = b_line + ',' + new_lines[1] + ',' + new_lines[2]
                        tgt_csv.writelines(b_line)
                        continue
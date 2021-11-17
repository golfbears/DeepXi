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
key_dic = {
    '000': ' ie2 k e1 j i4\n',
    '001': ' ie2 t ong2 x ve2\n',
    '002': 'd a3 k ai1 k ong1 t iao2\n',
    '003': 'g uan1 b i4 k ong1 t iao2\n',
    '004': 'sh eng1 g ao1 ii i1 d u4\n',
    '005': 'j iang4 d i1 ii i1 d u4\n',
    '006': 'z eng1 d a4 f eng1 s u4\n',
    '007': 'j ian3 x iao3 f eng1 s u4\n',
    '008': 'l ai2 d ian3 ii in1 vv ve4\n',
    '009': 't ing2 zh ix3 ii in1 vv ve4\n'
}

if __name__ == '__main__':

    home_dir = '/home/devpath/datasets/speech_corpus/shellshell'
    source_dir = os.path.join(home_dir,'shell_train_noisy')
    csv_dir = os.path.join(home_dir,'aishell-train-initial-final-asr-lexicon.csv')
    tgt_csv = os.path.join(home_dir,'shell-train-noisy-initial-final-asr-lexicon.csv')


    source_dir = os.path.join(home_dir, current_dir)
    src_list = os.listdir(source_dir)
    print(key_dic['000'])
    with open(tgt_csv, 'w') as tgt_csv:
        with open(csv_dir, 'r') as csv_dict:
            head = csv_dict.readline()
            tgt_csv.writelines(head)

            for fi in src_list:

                key = fi.split('_')[1]
                current_path = os.path.join(current_dir, fi)
                len = os.path.getsize(os.path.join(source_dir, fi))
                b_line = current_path + ',' + str(len) + ',' + key_dic[key]
                tgt_csv.writelines(b_line)

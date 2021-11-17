import numpy as np
from scipy import stats
import decimal, math
from soundfile import SoundFile, SEEK_END
import os, sys
from glob import glob
import librosa
from hybrid.phoneme import Phoneme
from hybrid.phonemes_utils import extract_phonemes_index, read_wav
from glob import glob
import librosa
import shutil
from tqdm import tqdm


def ensures_dir(directory: str):
    if len(directory) > 0 and not os.path.exists(directory):
        os.makedirs(directory)



def find_files(directory, ext='wav'):
    return sorted(glob(directory + '/**/*.{ext}', recursive=True))


if __name__ == '__main__':

    ph_dir = '/home/ml/speech-aligner/egs/cn_phn/data/shell/train_labels/ph_labels_epoch616'
    ph_infer_dir = '/home/devpath/datasets/aidatatang/train_labels/ph_label_infer'

    py_2_if_analyzer = Phoneme('/home/devpath/golfbears/DeepXi/hybrid', 'initialfinal2phoneme-lexicon.txt')
    # for wv in wv_fs:
    wv_fs = os.listdir(ph_dir)
    num_labels = len(wv_fs)
    for i, wv in tqdm(enumerate(wv_fs), desc='test', total=num_labels):
        with open(os.path.join(ph_dir, wv), 'r') as f_ph:
            i_read = f_ph.readlines()
            labels = [py_2_if_analyzer.tkn_dict.entry2Index[i.strip()] for i in i_read]
            if os.path.exists(os.path.join(ph_infer_dir, wv)):
                with open(os.path.join(ph_infer_dir, wv), 'r') as f_ph_infer:
                    i_read_infer = f_ph_infer.readlines()
                    if len(i_read) != len(i_read_infer):
                        print('ph:'+str(len(i_read)) + ' vs ph_infer:' + str(len(i_read_infer)))
                    else:
                        same_cnt = 0
                        whole_len = len(i_read_infer)
                        for idx in range(whole_len):
                            if i_read[idx] == i_read_infer[idx]:
                                same_cnt +=1
                        print(wv + ' has ' + str(same_cnt/whole_len) + ' similarity !')


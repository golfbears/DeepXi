import numpy as np
from scipy import stats
import decimal, math
from soundfile import SoundFile, SEEK_END
import os, sys
from glob import glob
import librosa

def read_wav(path):
	"""
	Read .wav file.

	Argument/s:
		path - absolute path to save .wav file.

	Returns:
		wav - waveform.
		f_s - sampling frequency.
	"""
	'''
	try: wav, f_s = sf.read(path, dtype='int16')
	except TypeError: f_s, wav = sf.read(path)
	return wav, f_s
	'''
	audio, sr = librosa.load(path, sr=16000, mono=True, dtype=np.float32)
	audio_int = audio*32767
	audio_int = audio_int.astype(np.int16)
	return audio_int, sr

def extract_phonemes_index():
    phonemes_dict_dir = '/home/devpath/golfbears/speech-aligner/egs/cn_phn/res/phones.txt'
    with open(phonemes_dict_dir, 'r') as f_phonemes_dict:
        phonemes_dict = f_phonemes_dict.readlines()[1:201]
    phonemes_dir = '/home/devpath/ml/speech-aligner/egs/cn_phn/data/analysis_MoG/gaussian_shellttd_aidatattd_thchs30'

    f_list = os.listdir(phonemes_dir)
    phonemes = []
    for i, phoneme in enumerate(phonemes_dict):
        #if phoneme.__contains__('sil') or phoneme.__contains__('$0'):
        if phoneme.__contains__('$0'):
            continue
        pho = phoneme.split(' ')[0]
        print('############################ check phoneme: No.' + str(i) + '-' + pho)
        available = 0
        for file in f_list:
            if not file.__contains__(pho):
                continue
            available = 1
            phonemes.append(pho)
            break
        #phonemes.append(pho)
        if available == 0:
            print('not available phoneme: No.' + str(i) + '-' + pho)
            continue
    phonemes.append('sil')
    return phonemes

def extract_label_from_phonemes_index(file_name, phonemes, f_len ,ali='/home/devpath/datasets/data/out.ali', phonemes_dict_dir = '/home/devpath/datasets/shellcsv/dev_labels'):

    starts = []
    stops = []
    idxs = []
    available = 0
    key = file_name.split('.')[0]
    with open(ali, 'r') as f_ali:
        for i, a_line in enumerate(f_ali):
            if a_line.__contains__(key):
                available = 1
                while True:
                    phoneme_line = f_ali.readline()
                    if phoneme_line == u'.\n':
                        break
                    else:
                        tst_lbs = np.zeros(1, int)
                        align_info = phoneme_line.strip().split(' ')
                        #current_phoneme_dir = os.path.join(phonemes_dir, align_info[2].strip())
                        start_offset = int(np.ceil(float(align_info[0].strip()) * 100))  # 10 ms per frames
                        stop_offset = int(np.floor(float(align_info[1].strip()) * 100))
                        pho = align_info[2].strip()
                        if (pho == '$0'):continue#pho = 'sil'
                        idx = phonemes.index(pho)
                        starts.append(start_offset)
                        stops.append(stop_offset)
                        idxs.append(idx)

                break
    if available == 1:
        labels = np.zeros(1, int)
        for j, id in enumerate(idxs[:-1]):
            labels = np.concatenate((labels, [idxs[j] for u in range(stops[j]-starts[j]+1)]))
            starts[j+1] = stops[j]+1
        labels = np.concatenate((labels, [idxs[-1] for u in range(stops[-1] - starts[-1]+1)]))
        if len(labels)-1 < f_len:
            labels = np.concatenate((labels, [phonemes[-1] for u in range(f_len - len(labels) + 1)]))
        np.save(os.path.join(phonemes_dict_dir, key + '_' + 'label'), labels[1:])
    pass
def find_files(directory, ext='wav'):

    return sorted(glob(directory + '/**/*.{ext}', recursive=True))


def simple_read_label(key, label_dir = '/home/devpath/datasets/data/labels'):
    npy_f = os.path.join(label_dir, key.split('.')[0]+'_label.npy')
    if os.path.exists(npy_f):
        return np.load(npy_f)
    else:
        return np.zeros(1, int)

def simple_read_label_new(key, label_dir = '/home/devpath/datasets/data/labels/ph_labels'):
    l_f = os.path.join(label_dir, key.split('.')[0] + '.csv')
    if os.path.exists(l_f):
        with open(l_f, 'r') as f_ph:
            labels = f_ph.readlines()
            return labels
    else: return []

if __name__ == '__main__':
    """
    phonemes_dict_dir = '/home/devpath/golfbears/speech-aligner/egs/cn_phn/res/phones.txt'
    wav_dir = '/home/devpath/datasets/speech_corpus/shellshell/data_aishell/wav/dev'
    ali_f = '/home/devpath/datasets/shellcsv/shell_dev_out.ali'
    phonemes_dict_dir = '/home/devpath/datasets/shellcsv/dev_labels'
    wv_fs = find_files(wav_dir)
    """
    phonemes_dict_dir = '/home/devpath/golfbears/speech-aligner/egs/cn_phn/res/phones.txt'
    wav_dir = '/home/devpath/datasets/thchs30-phoneme/data/wav'
    ali_f = '/home/devpath/datasets/thchs30-phoneme/data/out.ali'
    phonemes_dict_dir = '/home/devpath/datasets/thchs30-phoneme/data/labels'
    wv_fs = os.listdir(wav_dir)

    phonemes = extract_phonemes_index()
    ph_dict_dir = '/home/devpath/datasets/hybrid_phonemes_dict.bin'
    with open(ph_dict_dir, 'w') as f_dict:
        for i in phonemes:
            f_dict.writelines(i+'\n')
    with open(ph_dict_dir, 'r') as f_dict:
        cek = f_dict.readlines()
        pass
    for wv in wv_fs:
        #wv = wv.split('/')[-1]
        f = SoundFile(os.path.join(wav_dir, wv))
        wav_len = f.seek(0, SEEK_END)
        if wav_len == -1:
            wav, _ = read_wav(os.path.join(wav_dir, wv))
            wav_len = len(wav)
            if wav_len == -1: continue
        f_len = int(np.ceil(wav_len/160))


        extract_label_from_phonemes_index(wv, phonemes, f_len, ali_f, phonemes_dict_dir)
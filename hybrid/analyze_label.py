
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
def extract_label_from_phonemes_index(file_name, phonemes, f_len ,ali, phonemes_dict_dir, if_2_ph_analyzer):
    py_labels_dir = os.path.join(phonemes_dict_dir, 'py_label')
    ph_labels_dir = os.path.join(phonemes_dict_dir, 'ph_label')
    # wv = wv.split('/')[-1]
    #ensures_dir(py_labels_dir)
    #shutil.rmtree(py_labels_dir)
    #os.mkdir(py_labels_dir)
    #ensures_dir(ph_labels_dir)
    #shutil.rmtree(ph_labels_dir)
    #os.mkdir(ph_labels_dir)

    starts = []
    stops = []
    idxs = []
    available = 0
    pho_raw = []
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
                        pho_raw.append(pho)
                        idx = phonemes.index(pho)
                        starts.append(start_offset)
                        stops.append(stop_offset)
                        idxs.append(idx)

                break
    if available == 1:
        lenOfQue = len(pho_raw)
        i = 0
        while i < lenOfQue:
            if pho_raw[i] in ['sil']:
                i += 1
                continue
            elif pho_raw[i] in ['$0']:
                pho_raw[i] = 'sil'
                i += 1
                continue
            elif pho_raw[i].split('_')[0] in ['a', 'ao', 'an', 'ang', 'ai', 'e', 'ei', 'en', 'eng', 'o',  'ou', 'er']:
                i += 1
                continue
            elif pho_raw[i] in ['y']:
                if pho_raw[i+1].split('_')[0] in ['v', 've', 'vn', 'van']:
                    pho_raw[i] = 'v_0'
                elif pho_raw[i+1].split('_')[0] in ['u', 'ue', 'un', 'uan']:
                    pho_raw[i] = 'v_0'
                    pho_raw[i + 1] = pho_raw[i+1].replace('u','v')
                elif pho_raw[i+1].split('_')[0] in ['i', 'in', 'ing']:
                    pho_raw[i] = 'i_0'
                elif pho_raw[i+1].split('_')[0] in ['o']:
                    pho_raw[i] = 'i_0'
                    pho_raw[i + 1] = pho_raw[i+1].replace('o','iou')
                else:
                    pho_raw[i] = 'i_0'
                    pho_raw[i + 1] = pho_raw[i + 1].replace(pho_raw[i + 1], 'i'+pho_raw[i + 1])
                i += 2
                continue
            elif pho_raw[i] in ['w']:
                pho_raw[i] = 'u_0'
                if pho_raw[i + 1].split('_')[0] not in ['u']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace(pho_raw[i + 1], 'u'+pho_raw[i + 1])
                i += 2
                continue
            elif pho_raw[i] in ['j','q', 'x']:
                if pho_raw[i + 1].split('_')[0] in ['u', 'ue', 'un', 'uan']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'v')
                elif pho_raw[i + 1].split('_')[0] in ['iu']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'ou')
                i += 2
                continue
            elif pho_raw[i] in ['zh','ch','sh','r']:
                if pho_raw[i + 1].split('_')[0] in ['i']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('i', 'ix')
                elif pho_raw[i + 1].split('_')[0] in ['un', 'ui']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'ue')
                i += 2
                continue
            elif pho_raw[i] in ['z','c','s']:
                if pho_raw[i + 1].split('_')[0] in ['i']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('i', 'iy')
                elif pho_raw[i + 1].split('_')[0] in ['un', 'ui']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'ue')
                i += 2
                continue
            elif pho_raw[i] in ['b','p','m', 'f', 'n' ]:
                if pho_raw[i + 1].split('_')[0] in ['iu']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('i', 'io')
                i += 2
                continue
            elif pho_raw[i] in [ 'd', 't', 'l',  'g', 'k', 'h']:
                if pho_raw[i + 1].split('_')[0] in ['un', 'ui']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'ue')
                elif pho_raw[i + 1].split('_')[0] in ['iu']:
                    pho_raw[i + 1] = pho_raw[i + 1].replace('u', 'ou')
                i += 2
                continue
            else:
                print('abnormal initial_final analysis: '+ pho_raw[i] )
                pass

        labels = np.zeros(1, int)
        labels_py = []
        labels_ph = []
        for j, id in enumerate(idxs[:-1]):
            labels = np.concatenate((labels, [idxs[j] for u in range(stops[j]-starts[j]+1)]))
            [labels_py.append(pho_raw[j]) for u in range(stops[j]-starts[j]+1)]
            if pho_raw[j] in ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'i_0', 'i_1', 'i_2', 'i_3', 'i_4',
                              'ix_0', 'ix_1', 'ix_2', 'ix_3', 'ix_4', 'iy_0', 'iy_1', 'iy_2', 'iy_3', 'iy_4', 'o_0', 'o_1', 'o_2', 'o_3', 'o_4',
                              'u_0', 'u_1', 'u_2', 'u_3', 'u_4', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'er_0', 'er_1', 'er_2', 'er_3', 'er_4']:
                trans = pho_raw[j].replace('_', '')
                [labels_ph.append(trans) for u in range(stops[j] - starts[j] + 1)]
            elif pho_raw[j] in ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'zh', 'ch', 'sh', 'r']:
                [labels_ph.append(pho_raw[j]) for u in range(stops[j] - starts[j] + 1)]
            elif pho_raw[j] in ['ai_0', 'ai_1', 'ai_2', 'ai_3', 'ai_4', 'ao_0', 'ao_1', 'ao_2', 'ao_3', 'ao_4', 'an_0', 'an_1', 'an_2', 'an_3', 'an_4',
                                'ei_0', 'ei_1', 'ei_2', 'ei_3', 'ei_4', 'ou_0', 'ou_1', 'ou_2', 'ou_3', 'ou_4', 'en_0', 'en_1', 'en_2', 'en_3', 'en_4',
                                'ia_0', 'ia_1', 'ia_2', 'ia_3', 'ia_4', 'ie_0', 'ie_1', 'ie_2', 'ie_3', 'ie_4', 'in_0', 'in_1', 'in_2', 'in_3', 'in_4',
                                'ua_0', 'ua_1', 'ua_2', 'ua_3', 'ua_4', 'uo_0', 'uo_1', 'uo_2', 'uo_3', 'uo_4', 've_0', 've_1', 've_2', 've_3', 've_4',
                                'vn_0', 'vn_1', 'vn_2', 'vn_3', 'vn_4', 'ang_0', 'ang_1', 'ang_2', 'ang_3', 'ang_4', 'eng_0', 'eng_1', 'eng_2', 'eng_3', 'eng_4',
                                'ing_0', 'ing_1', 'ing_2', 'ing_3', 'ing_4', 'ong_0', 'ong_1', 'ong_2', 'ong_3', 'ong_4']:
                trans = pho_raw[j].split('_')
                ph_queue = if_2_ph_analyzer.pinyin[trans[0]].split(' ')
                lens = stops[j]-starts[j]+1
                [labels_ph.append(ph_queue[0] + trans[1]) for u in range(int(lens*0.3))]
                [labels_ph.append(ph_queue[1] + trans[1]) for u in range(lens-int(lens * 0.3))]
                pass
            elif pho_raw[j] in ['ian_0', 'ian_1', 'ian_2', 'ian_3', 'ian_4', 'iao_0', 'iao_1', 'iao_2', 'iao_3', 'iao_4', 'iang_0', 'iang_1', 'iang_2', 'iang_3', 'iang_4',
                                'iong_0', 'iong_1', 'iong_2', 'iong_3', 'iong_4', 'iou_0', 'iou_1', 'iou_2', 'iou_3', 'iou_4', 'uai_0', 'uai_1', 'uai_2', 'uai_3', 'uai_4',
                                'uang_0', 'uang_1', 'uang_2', 'uang_3', 'uang_4', 'uei_0', 'uei_1', 'uei_2', 'uei_3', 'uei_4', 'uen_0', 'uen_1', 'uen_2', 'uen_3', 'uen_4',
                                'uan_0', 'uan_1', 'uan_2', 'uan_3', 'uan_4', 'van_0', 'van_1', 'van_2', 'van_3', 'van_4', 'ueng_0', 'ueng_1', 'ueng_2', 'ueng_3', 'ueng_4']:
                trans = pho_raw[j].split('_')
                ph_queue = if_2_ph_analyzer.pinyin[trans[0]].split(' ')
                lens = stops[j]-starts[j]+1
                [labels_ph.append(ph_queue[0] + '0') for u in range(int(lens*0.3))]
                [labels_ph.append(ph_queue[1] + trans[1]) for u in range(int(lens * 0.4))]
                [labels_ph.append(ph_queue[2] + trans[1]) for u in range(lens-int(lens * 0.3)-int(lens * 0.4))]
                pass
            elif pho_raw[j] in ['sil']:
                [labels_ph.append('|') for u in range(stops[j] - starts[j] + 1)]
                pass
            else:
                print('abnormal phonemes: ' + pho_raw[j])
                pass
            starts[j+1] = stops[j]+1
        labels = np.concatenate((labels, [idxs[-1] for u in range(stops[-1] - starts[-1]+1)]))
        [labels_py.append(pho_raw[-1]) for u in range(stops[-1] - starts[-1] + 1)]
        if pho_raw[-1] in ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'i_0', 'i_1', 'i_2',
                          'i_3', 'i_4',
                          'ix_0', 'ix_1', 'ix_2', 'ix_3', 'ix_4', 'iy_0', 'iy_1', 'iy_2', 'iy_3', 'iy_4', 'o_0', 'o_1',
                          'o_2', 'o_3', 'o_4',
                          'u_0', 'u_1', 'u_2', 'u_3', 'u_4', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'er_0', 'er_1', 'er_2',
                          'er_3', 'er_4']:
            trans = pho_raw[-1].replace('_', '')
            [labels_ph.append(trans) for u in range(stops[-1] - starts[-1] + 1)]
        elif pho_raw[-1] in ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'zh',
                            'ch', 'sh', 'r']:
            [labels_ph.append(pho_raw[-1]) for u in range(stops[-1] - starts[-1] + 1)]
        elif pho_raw[-1] in ['ai_0', 'ai_1', 'ai_2', 'ai_3', 'ai_4', 'ao_0', 'ao_1', 'ao_2', 'ao_3', 'ao_4', 'an_0',
                            'an_1', 'an_2', 'an_3', 'an_4',
                            'ei_0', 'ei_1', 'ei_2', 'ei_3', 'ei_4', 'ou_0', 'ou_1', 'ou_2', 'ou_3', 'ou_4', 'en_0',
                            'en_1', 'en_2', 'en_3', 'en_4',
                            'ia_0', 'ia_1', 'ia_2', 'ia_3', 'ia_4', 'ie_0', 'ie_1', 'ie_2', 'ie_3', 'ie_4', 'in_0',
                            'in_1', 'in_2', 'in_3', 'in_4',
                            'ua_0', 'ua_1', 'ua_2', 'ua_3', 'ua_4', 'uo_0', 'uo_1', 'uo_2', 'uo_3', 'uo_4', 've_0',
                            've_1', 've_2', 've_3', 've_4',
                            'vn_0', 'vn_1', 'vn_2', 'vn_3', 'vn_4', 'ang_0', 'ang_1', 'ang_2', 'ang_3', 'ang_4',
                            'eng_0', 'eng_1', 'eng_2', 'eng_3', 'eng_4',
                            'ing_0', 'ing_1', 'ing_2', 'ing_3', 'ing_4', 'ong_0', 'ong_1', 'ong_2', 'ong_3', 'ong_4']:
            trans = pho_raw[-1].split('_')
            ph_queue = if_2_ph_analyzer.pinyin[trans[0]].split(' ')
            lens = stops[-1] - starts[-1] + 1
            [labels_ph.append(ph_queue[0] + trans[1]) for u in range(int(lens * 0.3))]
            [labels_ph.append(ph_queue[1] + trans[1]) for u in range(lens - int(lens * 0.3))]
            pass
        elif pho_raw[-1] in ['ian_0', 'ian_1', 'ian_2', 'ian_3', 'ian_4', 'iao_0', 'iao_1', 'iao_2', 'iao_3', 'iao_4',
                            'iang_0', 'iang_1', 'iang_2', 'iang_3', 'iang_4',
                            'iong_0', 'iong_1', 'iong_2', 'iong_3', 'iong_4', 'iou_0', 'iou_1', 'iou_2', 'iou_3',
                            'iou_4', 'uai_0', 'uai_1', 'uai_2', 'uai_3', 'uai_4',
                            'uang_0', 'uang_1', 'uang_2', 'uang_3', 'uang_4', 'uei_0', 'uei_1', 'uei_2', 'uei_3',
                            'uei_4', 'uen_0', 'uen_1', 'uen_2', 'uen_3', 'uen_4',
                            'uan_0', 'uan_1', 'uan_2', 'uan_3', 'uan_4', 'van_0', 'van_1', 'van_2', 'van_3', 'van_4',
                            'ueng_0', 'ueng_1', 'ueng_2', 'ueng_3', 'ueng_4']:
            trans = pho_raw[-1].split('_')
            ph_queue = if_2_ph_analyzer.pinyin[trans[0]].split(' ')
            lens = stops[-1] - starts[-1] + 1
            [labels_ph.append(ph_queue[0] + '0') for u in range(int(lens * 0.3))]
            [labels_ph.append(ph_queue[1] + trans[1]) for u in range(int(lens * 0.4))]
            [labels_ph.append(ph_queue[2] + trans[1]) for u in range(lens - int(lens * 0.3) - int(lens * 0.4))]
            pass
        elif pho_raw[-1] in ['sil']:
            [labels_ph.append('|') for u in range(stops[-1] - starts[-1] + 1)]
            pass
        else:
            print('abnormal phonemes: ' + pho_raw[-1])
            pass
        if len(labels)-1 < f_len:
            labels = np.concatenate((labels, [phonemes[-1].index for u in range(f_len - len(labels) + 1)]))
            [labels_py.append(pho_raw[-1]) for u in range(f_len - len(labels) + 1)]
            [labels_ph.append('|') for u in range(f_len - len(labels) + 1)]
        for i in range(len(labels_ph)):
            if labels_ph[i].__contains__('0'):
                labels_ph[i]=labels_ph[i].replace('0','5')
        if len(labels_ph) > 1:
            #np.save(os.path.join(phonemes_dict_dir, key + '_' + 'label'), labels[1:])
            with open(os.path.join(ph_labels_dir, key+'.csv'), 'w') as f_ph:
                for i in labels_ph:
                    f_ph.writelines(i + '\n')
            with open(os.path.join(py_labels_dir, key+'.csv'), 'w') as f_py:
                for i in labels_py:
                    f_py.writelines(i + '\n')
        else:
            print('abnormal analyze:' + key)
    pass
def find_files(directory, ext='wav'):

    return sorted(glob(directory + '/**/*.{ext}', recursive=True))


if __name__ == '__main__':
    '''
    base_dir = '/home/devpath/ml/speech-aligner/egs/cn_phn/data/shell'
    test_r1 = os.path.join(base_dir,'test_labels_round1')
    test_r2 = os.path.join(base_dir,'test_labels_round2')
    wv_fs1 = os.listdir(test_r1)
    wv_fs2 = os.listdir(test_r2)
    prj_dir = '/home/devpath/golfbears/DeepXi/hybrid'
    if_2_ph = 'initialfinal2phoneme-lexicon.txt'
    py_2_if = 'pinyin2initialfinal-lexicon.txt'
    if_2_ph_analyzer = Phoneme(os.path.join(prj_dir, if_2_ph))
    py_2_if_analyzer = Phoneme(os.path.join(prj_dir, py_2_if))
    '''
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

    py_labels_dir = os.path.join(phonemes_dict_dir, 'py_label')
    ph_labels_dir = os.path.join(phonemes_dict_dir, 'ph_label')
   
    ensures_dir(py_labels_dir)
    shutil.rmtree(py_labels_dir)
    os.mkdir(py_labels_dir)
    ensures_dir(ph_labels_dir)
    shutil.rmtree(ph_labels_dir)
    os.mkdir(ph_labels_dir)
    phonemes = extract_phonemes_index()
    ph_dict_dir = '/home/devpath/datasets/hybrid_phonemes_dict.bin'
    prj_dir = '/home/devpath/golfbears/DeepXi/hybrid'
    if_2_ph = 'initialfinal2phoneme-lexicon.txt'
    py_2_if = 'pinyin2initialfinal-lexicon.txt'
    if_2_ph_analyzer = Phoneme(os.path.join(prj_dir, if_2_ph))
    #py_2_if_analyzer = Phoneme(os.path.join(prj_dir, py_2_if))
    #for wv in wv_fs:
    num_labels = len(wv_fs)
    for i, wv in  tqdm(enumerate(wv_fs), desc='test', total=num_labels):
        #wv = wv.split('/')[-1]
        f = SoundFile(os.path.join(wav_dir, wv))
        wav_len = f.seek(0, SEEK_END)
        if wav_len == -1:
            wav, _ = read_wav(os.path.join(wav_dir, wv))
            wav_len = len(wav)
            if wav_len == -1: continue
        f_len = int(np.ceil(wav_len/160))


        extract_label_from_phonemes_index(wv, phonemes, f_len, ali_f, phonemes_dict_dir, if_2_ph_analyzer)

    '''
    for i, f1  in enumerate(wv_fs1):
        l1 = np.load(os.path.join(test_r1,f1))
        l2 = np.load(os.path.join(test_r2,f1))
        print(np.sum(l1==l2)/len(l1))
    '''

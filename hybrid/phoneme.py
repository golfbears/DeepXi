import re
import os
import numpy as np
class Dictionary:
    def __init__(self, d1: dict, d2: dict):
        self.entry2Index = d1
        self.index2Entry = d2

class Phoneme:
    def __init__(self, phoneme_lexicon:str):
        pinyin, phoneme_d1, phoneme_d2 = self._parse_phoneme_lexicon(phoneme_lexicon)
        self.pinyin = pinyin
        self.tkn_dict = Dictionary(phoneme_d1, phoneme_d2)
        self.get_target_ids = self.get_from_initial_final2phonemes

    def _print_dictionary(self, d, name):

        print('################################### ', name, ' dictionary*************************************:')
        i = 0
        for a in d:
            print(a, ':', d[a], end='|\t')
            i += 1
            if i == 20:
                print('')
                i = 0
        print('\n################################### end ***********************************************')

    def _parse_phonemes_from_lines(self, lines, callback):
        d0 = {}  # pinyin -> initials/finals
        d1 = {}  # initials/finals -> idx
        d2 = {}  # idx -> initials/finals
        idx = 0
        d1['|'] = idx
        d2[idx] = '|'

        idx += 1
        for line in lines:

            items = line.split('|')
            pinyin = items[0].strip()
            if pinyin in ['aa','oo','ee','ii','uu','vv']:
                continue
            initials_finals = items[1].strip()
            if not d0.__contains__(pinyin):
                d0[pinyin] = initials_finals

            phonemes = initials_finals.split()
            if len(phonemes) == 1:
                if phonemes[0] in ['b', 'p', 'd', 't', 'j', 'q', 'x', 'm', 'n', 'f', 'l', 'g', 'k', 'h', 'z', 'c', 's', 'r', 'zh', 'ch', 'sh']:
                    if not d1.__contains__(phonemes[0]):
                        d1[phonemes[0]] = idx
                        if not d2.__contains__(idx):
                            d2[idx] = phonemes[0]
                        idx += 1
                #elif phonemes[0] == 'n':#n -- n5
                #    continue
                else:#[a, i, ix, iy, o, e, u, v, er]
                    for tone in ['1', '2', '3', '4', '5']:
                        if not d1.__contains__(phonemes[0] + tone):
                            d1[phonemes[0] + tone] = idx
                            if not d2.__contains__(idx):
                                d2[idx] = phonemes[0] + tone
                            idx += 1
            else:# len(phonemes) == 2 or 3:
                for phoneme in phonemes:
                    for tone in ['1', '2', '3', '4', '5']:
                        if not d1.__contains__(phoneme + tone):
                            d1[phoneme + tone] = idx
                            if not d2.__contains__(idx):
                                d2[idx] = phoneme + tone
                            idx += 1

        d1['*'] = idx
        d2[idx] = '*'
        self._print_dictionary(d0, 'initial and finals - phonemes')
        self._print_dictionary(d1, 'phonemes token')

        return d0, d1, d2

    def _parse_initials_finals_from_lines(self, lines, callback):
        d0 = {} # pinyin -> initials/finals
        d1 = {} # initials/finals -> idx
        d2 = {} # idx -> initials/finals
        idx = 0
        d1['|'] = idx
        d2[idx] = '|'

        idx += 1
        for line in lines:

            items = line.split('|')
            pinyin = items[0].strip()
            initials_finals = items[1].strip()
            if not d0.__contains__(pinyin):
                d0[pinyin] = initials_finals

            phonemes = initials_finals.split()
            if not d1.__contains__(phonemes[0]):
                d1[phonemes[0]] = idx
                if not d2.__contains__(idx):
                    d2[idx] = phonemes[0]
                idx += 1
            for tone in  ['1','2','3','4','5']:
                if not d1.__contains__(phonemes[1]+tone):
                    d1[phonemes[1]+tone] = idx
                    if not d2.__contains__(idx):
                        d2[idx] = phonemes[1]+tone
                    idx += 1

        d1['*'] = idx
        d2[idx] = '*'
        self._print_dictionary(d0, 'pinyin')
        self._print_dictionary(d1, 'initial and final token')

        return d0, d1, d2

    def _parse_phoneme_lexicon(self, phoneme_lexicon: str):
        def token_callback(s):
            return s

        with open(phoneme_lexicon, 'r') as f:
            lines = f.readlines();
            if len(lines) == 65:# phonemes
                return self._parse_phonemes_from_lines(lines, token_callback)
            elif len(lines) == 474:#initial-final
                return self._parse_initials_finals_from_lines(lines, token_callback)


    def get_from_pinyin2initial_final(self, sentence):
        ids = []
        pinyin = sentence.split()
        for t in pinyin:

            t2 = re.sub('[1/2/3/4/5]', '', t)
            tone = t[-1]
            if tone not in ['1','2','3','4','5']:
                tone = '5'
            if self.pinyin.__contains__(t2):

                initial_final = self.pinyin[t2]
                initial_final = initial_final.split()
                if self.tkn_dict.entry2Index.__contains__(initial_final[0]):
                    ids.append(self.tkn_dict.entry2Index[initial_final[0]])
                else:
                    #if not initial_final[0] in ['aa', 'oo', 'ee', 'ii', 'uu', 'vv']:
                    ids.append(self.unknown_token_idx)
                    print('should not run here!:', initial_final, initial_final[0])
                if self.tkn_dict.entry2Index.__contains__(initial_final[1]+tone):
                    ids.append(self.tkn_dict.entry2Index[initial_final[1]+tone])
                else:
                    ids.append(self.unknown_token_idx)
                    print('should not run here!:', initial_final, initial_final[1])
                #ids.append(self.tkn_dict.entry2Index['|'])
            else:
                print('should not run here!:', t2)
        return ids

    def get_from_initial_final2initial_final(self, sentence):
        ids = []
        initial_final = sentence.split()
        for t in initial_final:
            if self.tkn_dict.entry2Index.__contains__(t):
                ids.append(self.tkn_dict.entry2Index[t])
            else:
                # if not initial_final[0] in ['aa', 'oo', 'ee', 'ii', 'uu', 'vv']:
                ids.append(self.unknown_token_idx)
                print('should not run here!:', sentence, t)
        return ids

    def get_from_initial_final2phonemes(self, sentence):
        ids = []
        initial_final = sentence.split()
        if self.tkn_dict.entry2Index.__contains__('|'):
            ids.append(self.tkn_dict.entry2Index['|'])
        else:
            print('should not run here Initial!:', initial_final, sentence)

        for t in initial_final:
            if t in  ['aa','oo','ee','ii','uu','vv']:
                continue
            a = t[-1]
            if a not in ['1','2','3','4','5']:
                if t in self.pinyin:
                    initial = self.pinyin[t]
                    if initial == 'n':
                        if self.tkn_dict.entry2Index.__contains__('n5'):
                            ids.append(self.tkn_dict.entry2Index['n5'])
                        else:
                            print('should not run here Initial!:', initial, t)
                    elif self.tkn_dict.entry2Index.__contains__(initial):
                        ids.append(self.tkn_dict.entry2Index[initial])
                    else:
                        print('should not run here Initial!:', initial, t)
            else:
                final= t[:-1]
                if final in self.pinyin:
                    final_phonems = self.pinyin[final]
                    final_phonems = final_phonems.split()
                    if len(final_phonems) == 3:
                        final_head = final_phonems[0]
                        final_last = final_phonems[1:]
                        if self.tkn_dict.entry2Index.__contains__(final_head+'5'):
                            ids.append(self.tkn_dict.entry2Index[final_head+'5'])
                        else:
                            print('should not run here final_head!:', final_head, t)
                    else:
                        final_last = final_phonems

                    for pho in final_last:
                        if self.tkn_dict.entry2Index.__contains__(pho+a):
                            ids.append(self.tkn_dict.entry2Index[pho+a])
                        else:
                            print('should not run here Initial!:', pho, t)

                if self.tkn_dict.entry2Index.__contains__('|'):
                    ids.append(self.tkn_dict.entry2Index['|'])
                else:
                    print('should not run here Initial!:', pho, t)

        return ids
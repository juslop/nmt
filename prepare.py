#
#   Copyright 2019 Jussi Löppönen
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import re
import os
import deepdish as dd
import numpy as np
import random
from sklearn.utils import shuffle
from collections import OrderedDict


UNK = '<UNK>'
PAD = 0
NUMBER = '<NUMBER>'
START = '<START>'
EOL = '<EOL>'
CHUNK_SIZE = 100000
# single GPU limits embedding side
# DICTIONARY_MAX_LENGTH = 250000

# downloaded files in windows...
file_kwargs = {}
if os.name == 'nt':
    file_kwargs['encoding'] = 'utf-8'

def clean_line(l):
    # mark all numbers with NUMBER
    l = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", NUMBER, l.strip())
    # remove all words containing a number.
    l = re.sub("\S*\d\S*", "", l).strip()
    l = l.replace('*', '').replace('--', '').replace("'", "").replace("-", " - ").replace("`", "")
    # separate with space points, commas etc.
    l = " ".join(re.sub(r'([?(),.!\[\]\'\"#€$:;|<>%&/])', r' \1 ', l.lower()).split())
    # add end of line symbol to the end
    l += " " + EOL
    return l

def handle_bracket(l, brackets):
    index = l.find(brackets[0] + ' ')
    if index > -1 and len(l) > index + 2:
        l = l[:index + 1] + _close_bracket(l[index + 2:], brackets[1])
        if len(l) > index + 2:
            l = l[:index + 2] + handle_bracket(l[index + 2:], brackets)
    return l

def _close_bracket(l, bracket):
    index = l.find(' ' + bracket)
    if index > -1:
        return l[0: index] + l[index + 1:]
    return l

def _close_quote(l):
    index = l.find(' "')
    if index > 0:
        return l[:index] + l[index + 1:]
    return l

def handle_quotes(l):
    index = l.find('" ')
    if index > -1 and len(l) > index + 2:
        l = l[:index + 1] + _close_quote(l[index + 2:])
        index = l.find(' "', index + 2)
        if index > 0:
            l = l[:index + 1] + handle_quotes(l[index + 1:])
    return l

def process_line(sentence, dictionary, unk_index, sequence_len):
    '''
    convert words to indexes, pad and clip
    '''
    line = clean_line(sentence)
    line = [dictionary.get(word, unk_index) for word in line.split()]
    line = line[:sequence_len]
    if sequence_len > len(line):
        line += [PAD] * (sequence_len - len(line))
    return line

def post_process_line(l):
    #l = l.replace(EOL, "").rstrip()
    for m in [' ,', ' .', ' ?', ' !']:
        l = l.replace(m, m.lstrip())
    for brackets in [('<', '>'), ('[', ']'), ('(', ')')]:
        l = handle_bracket(l, brackets)
    l = handle_quotes(l)
    return l.capitalize()

def indexes_to_words(line_as_integers, index_to_word_map):
    '''
    converts integer sentence back to words
    '''
    return [index_to_word_map.get(i) for i in line_as_integers]

def get_unk_index(dictionary):
    return dictionary[UNK]

def get_start_index(dictionary):
    return dictionary[START]

def get_end_index(dictionary):
    return dictionary[EOL]

def create_dictionary(langs, filenames, skip_words_treshold, work_dir):
    '''
    collects words from file pairs to dictionary.
    Arguments:
    langs -- tuple of 2 languages to be translated
    filenames -- list of filepairs containing the texts
    work_dir -- where to store dictionary
    '''
    print("preparing dictionary")
    words = [{}, {}]
    for files in filenames:
        for i in range(2):
            print("processing file: {}".format(files[i]))
            with open(files[i], **file_kwargs) as fp:
                for line in fp:
                    sentence = clean_line(line).split()
                    for word in sentence:
                        word = word.strip()
                        cnt = words[i].get(word, 0)
                        words[i][word] = cnt + 1
    dct = {}
    indexes_to_words = {}
    for i in range(2):
        print("building dictionary for {}".format(langs[i]))
        words[i] = {k:v for k, v in words[i].items() if v >= skip_words_treshold}
        ordered = OrderedDict(sorted(words[i].items(), key=lambda x: x[1]))
        dict_list = list(ordered.keys()) #[:DICTIONARY_MAX_LENGTH]
        dict_list = sorted(dict_list)
        #add unknown word to vocabulary
        dict_list += [UNK, START, NUMBER, EOL]
        # for debugging
        with open("{}-dict.txt".format(langs[i]), "w", **file_kwargs) as f:
            for word in dict_list:
                f.write("%s\n" % word)
        dct[langs[i]] = dict(map(reversed, enumerate(dict_list, start=1)))
        #add pad to vocabulary
        dct[langs[i]][PAD] = 0
        indexes_to_words[langs[i]] = dict((v, k) for k, v in dct[langs[i]].items())
    print("size of {} dictionary: {}".format(langs[0], len(dct[langs[0]])))
    print("size of {} dictionary: {}".format(langs[1], len(dct[langs[1]])))
    for lang in langs:
        save_file = os.path.join(work_dir,
            "{}-dictionary.h5".format(lang))
        dd.io.save(save_file, 
            {'dictionary': dct[lang], 
            'index_to_word_map': indexes_to_words[lang]},
            compression=None)
        print("saved dictionary to {}".format(save_file))
    print("dictionary ready")

def read_dictionary(languages, work_dir):
    '''
    reads saved dictionaries
    Returns:
    dct -- for both languages dictionary of words and enumerated index
    indexes_to_words -- for both languages index to word conversion dictionary
    '''
    dct = {}
    index_to_word_map = {}
    for lang in languages:
        filename = os.path.join(work_dir,
            "{}-dictionary.h5".format(lang))
        loaded = dd.io.load(filename)
        dct[lang] = loaded['dictionary']
        index_to_word_map[lang] = loaded['index_to_word_map']
    return dct, index_to_word_map

def _read_text_chunk(fp):
    '''
    read a chunk or sentences from training data file
    lines -- list of word indexes
    '''
    CHUNK_SIZE = 100000
    lines = []
    for _ in range(CHUNK_SIZE):
        line = fp.readline()
        if not line:
            break
        lines.append(line)
    return lines

def _make_test_sets(arr, test_set_size):
    '''
    splits data to training and validation sets
    '''
    size = arr.shape[0]
    m = int(size*(1-test_set_size))
    return arr[:m, :], arr[m:, :]

def _file_names(lang, work_dir):
    return os.path.join(work_dir, "{}-indexes.npy".format(lang)), os.path.join(work_dir, "{}-test-sets.npy".format(lang))

def convert_words_to_indexes(langs, filenames, work_dir, Tx, Ty, test_set_size):
    '''
    read in the training material, convert words to indexes 
    '''
    print("converting words to indexes")
    dct, index_to_wordmap = read_dictionary(langs, work_dir)
    all_indexes = [np.empty((0, Tx), int),
        np.empty((0, Ty), int)]
    for files in filenames:
        for i, filename in enumerate(files):
            print("processing file: {}".format(filename))
            sequence_len = Tx if i == 0 else Ty
            unk_index = get_unk_index(dct[langs[i]])
            with open(files[i], **file_kwargs) as fp:
                while True:
                    lines = _read_text_chunk(fp)
                    if len(lines) == 0:
                        break
                    indexes = []
                    for sentence in lines:
                        indexes.append(process_line(sentence, dct[langs[i]],
                            unk_index, sequence_len))
                    assert len(indexes[0]) == sequence_len, "wrong size {}".format(len(indexes[0]))
                    all_indexes[i] = np.append(all_indexes[i], indexes, axis=0)
    assert all_indexes[0].shape[0] == all_indexes[1].shape[0],  "training data length mismatch"
    #test indexing
    assert index_to_wordmap["english"][dct["english"]["is"]] == "is", "mapping incorrect"
    all_indexes = shuffle(all_indexes[0], all_indexes[1])
    for i in range(2):
        data, tests = _make_test_sets(all_indexes[i], test_set_size)
        data_file, test_file = _file_names(langs[i], work_dir)
        np.save(data_file, data)
        np.save(test_file, tests)
        print("saved indexes to {}.npy {}.npy".format(data_file, test_file))
    print("word to index conversion done. {} lines of text".format(all_indexes[0].shape[0]))

def get_training_data(work_dir, languages):
    '''
    returns training data as numpy memmap on disk handles.
    swapping languages switches training order
    Returns:
    tuple X, X_test, Y, Y_test
    '''
    handles = []
    for lang in languages:
        data_file, test_file = _file_names(lang, work_dir)
        data = np.load(data_file)
        tests = np.load(test_file)
        handles.append([data, tests])
    X = handles[0][0]
    X_test = handles[0][1]
    Y = handles[1][0]
    Y_test = handles[1][1]
    return X, X_test, Y, Y_test

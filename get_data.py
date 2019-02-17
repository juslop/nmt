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

from tensorflow.keras.utils import get_file
import os, io
import json
import numpy as np

'''
Glove vectors:
https://fasttext.cc/docs/en/support.html

@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
'''

with open('config.json') as json_data_file:
    data = json.load(json_data_file)

EMBEDDING_SIZE = 300

def get_data():
    '''
    fetches language training data files from .keras cache or from net
    '''
    translation_files = data['translation_files']
    file_pairs = []
    for tgt in translation_files:
        path = get_file(
            tgt["fname"],
            origin=tgt["url"],
            cache_subdir=tgt["subdir"],
            extract=True)
        dirname = os.path.dirname(path)
        file_pairs.append([os.path.join(dirname, tgt["extracted"][i]) for i in range(2)])
    return file_pairs

def _get_vectors(lang):
    vf = data['word_vector_files'][lang]
    path = get_file(
        vf["fname"],
        origin=vf["url"],
        cache_subdir=vf["subdir"],
        extract=True
    ) 
    # in windows need manually to extract
    return os.path.join(os.path.dirname(path), vf["extracted"])

def _process_glove_vectors(filename):
    embeddings = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors
            if i % 250000 == 0:
                print("processed {} vectors".format(i))
    return embeddings

def _dictionary_to_glove_vector_map(language, dictionary, embeddings, work_dir):
    vocabulary_size = len(dictionary) + 1
    embedding_matrix = np.random.uniform(-1, 1, size=(vocabulary_size, EMBEDDING_SIZE))
    num_loaded = 0
    for w, i in dictionary.items():
        v = embeddings.get(w)
        if v is not None and i < vocabulary_size:
            embedding_matrix[i] = v
            num_loaded += 1
    print("found {} word vectors from glove dictionary for {}".format(num_loaded, language))
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix

def _embedding_matrix_path(language, work_dir):
    return os.path.join(work_dir, "{}-embeddings.npy".format(language))


def fetch_glove_vectors(languages, dictionaries, work_dir):
    '''
    fetches glove vectors for languages and builds embedding matrix for the previously
    buildt dictionary. Saves embedding matrixes to disk.
    '''
    for lang in languages:
        dictionary = dictionaries[lang]
        print("fetching glove vectors for {}".format(lang))
        path = _get_vectors(lang)
        embeddings = _process_glove_vectors(path)
        embedding_matrix = _dictionary_to_glove_vector_map(lang, dictionary, embeddings, work_dir)
        filename = _embedding_matrix_path(lang, work_dir)
        np.save(filename, embedding_matrix)
        print("saved {} embedding matrix to {}".format(embedding_matrix.shape, filename))


def get_embedding_matrix(language, work_dir):
    '''
    returns previously save numpy array embedding matrix
    '''
    return np.load(_embedding_matrix_path(language, work_dir))

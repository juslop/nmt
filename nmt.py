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

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
from models import Encoder, TrainingDecoder, GreedyDecoder, BeamSearchDecoder
from prepare import (read_dictionary, get_end_index, get_start_index)
from get_data import get_embedding_matrix

class NMTModel:
    def __init__(self, languages, Tx, Ty, num_layers, units, batch_size, work_dir, lr=0.001, beam_width=1):
        self.dictionary, self.index_to_word_map = read_dictionary(languages, work_dir)
        self.work_dir = work_dir
        self.languages = languages
        self.to_start_index = get_start_index(self.dictionary[self.languages[1]])
        self.from_embeddings = get_embedding_matrix(self.languages[0], self.work_dir)
        self.to_embeddings = get_embedding_matrix(self.languages[1], self.work_dir)
        self.to_vocabulary_size = self.to_embeddings.shape[0]
        self.to_end_index = get_end_index(self.dictionary[self.languages[1]])

        self.batch_size = batch_size
        self.beam_width = beam_width
        self.units = units
        self.num_layers = num_layers
        self.Tx = Tx # input sentence max length
        self.Ty = Ty # translated sentence max
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.lr = lr
        self._input = None
        self._outputs = None
        print("length of {} dictionary: {} words".format(languages[0], len(self.dictionary[languages[0]])))
        print("length of {} dictionary: {} words".format(languages[1], len(self.dictionary[languages[1]])))

    def _loss(self, logits, labels):
        # mask padded zeros out
        mask = tf.cast(labels > 0, tf.float32)
        return tf.contrib.seq2seq.sequence_loss(logits=logits, 
            targets=labels, 
            weights=mask, average_across_batch=False)

    def _backwards(self, loss):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        return self.optimizer.apply_gradients(zip(clipped_gradients, params))

    def _build(self, is_training):
        self.encoder = Encoder(self.from_embeddings.shape, self.num_layers, self.units, is_training=is_training)
        if is_training:
            self.decoder = TrainingDecoder(
                self.Ty, self.to_embeddings.shape, self.to_start_index, 
                self.to_end_index, self.to_vocabulary_size, 
                self.num_layers, self.units, self.batch_size)
        elif self.beam_width > 1:
            self.decoder = BeamSearchDecoder(
                self.beam_width, self.Ty, self.to_embeddings.shape, self.to_start_index, 
                self.to_end_index, self.to_vocabulary_size, 
                self.num_layers, self.units, self.batch_size)
        else:
            self.decoder = GreedyDecoder(
                self.Ty, self.to_embeddings.shape, self.to_start_index, 
                self.to_end_index, self.to_vocabulary_size, 
                self.num_layers, self.units, self.batch_size)

    def initialize_embeddings(self, sess):
        sess.run(self.encoder.from_embedding_init, feed_dict={self.encoder.from_embedding_placeholder: self.from_embeddings})
        sess.run(self.decoder.to_embedding_init, feed_dict={self.decoder.to_embedding_placeholder: self.to_embeddings})
        self.from_embeddings = None
        self.to_embeddings = None

    def build_graph(self, is_training=False):
        self._inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.Tx), name="inputs")
        self._build(is_training)
        if is_training:
            self._outputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.Ty), name='outputs')
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            encoder_outputs = self.encoder(self._inputs)
            self.encoder.log_to_tensorboard()
            outputs = self.decoder(encoder_outputs, self._outputs)
            loss = self._loss(outputs, self._outputs)
            self._train_ops = self._backwards(loss)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)
            self._cost = loss
        else:
            encoder_outputs = self.encoder(self._inputs)
            self._translation = self.decoder(encoder_outputs)

    def get_variables(self):
        trainable_variables = [v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if not "embeddings" in v.name]
        non_trainable_vars = [self.encoder.from_embeddings, self.decoder.to_embeddings]
        return trainable_variables, non_trainable_vars

    @property
    def cost(self):
        return self._cost

    @property
    def train_ops(self):
        return self._train_ops

    @property
    def translation(self):
        return self._translation

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

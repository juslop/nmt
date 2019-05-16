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

import numpy as np
import os, shutil
import time
import tensorflow as tf
import nltk
from nmt_model import NMTModel
from prepare import (read_dictionary, get_training_data, 
    get_unk_index, get_end_index, process_line, indexes_to_words,
    process_line, post_process_line, get_start_index)
from get_data import get_embedding_matrix


class RunNMT:
    def __init__(self, languages, Tx, Ty, num_layers, units, batch_size, work_dir, lr=0.001, beam_width=1):
        assert tf.test.is_gpu_available(), "This application requires GPU to run."
        self.checkpoint_dir = os.path.abspath('{}/{}-{}-checkpoints'.format(work_dir, languages[0], languages[1]))
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model.ckpt")
        self.weights_dir = 'weights/{}-{}'.format(languages[0], languages[1])
        self.weights_path = '{}/model.ckpt'.format(self.weights_dir)
        self.languages = languages
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.epoch_steps = None
        self.model = NMTModel(languages, Tx, Ty, num_layers, units, batch_size, work_dir, lr=lr, beam_width=beam_width)

    def create_dataset(self, test=False):
        X, _, Y, _ = get_training_data(self.work_dir, self.languages)
        self.epoch_steps = X.shape[0] // self.batch_size
        print("dataset length: {} lines".format(X.shape[0]))

        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(X.shape[0])
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def create_validationset(self, test=False):
        _, X_test, _, Y_test = get_training_data(self.work_dir, self.languages)
        print("test_set {} lines".format(X_test.shape[0]))

        dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(X_test.shape[0])
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def train(self, epochs, load_checkpoint=False):
        print_interval = 500
        use_tensorboard = os.environ.get("USE_TENSORBOARD") is not None

        self.model.build_graph(is_training=True)
        dataset = self.create_dataset()

        for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v.name, v.shape)

        if not load_checkpoint:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
        else:
            trainable_variables, non_trainable_vars = self.model.get_variables()
            saver = tf.train.Saver(var_list=trainable_variables)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            if use_tensorboard:
                summary_writer = tf.summary.FileWriter(os.path.join(self.work_dir, "tensorboard"), graph=sess.graph)
            if not load_checkpoint:
                sess.run(init)
            else:
                sess.run(tf.variables_initializer(non_trainable_vars))
                saver.restore(sess, self.checkpoint_path)
            self.model.initialize_embeddings(sess)

            for epoch in range(epochs):
                iterator = dataset.make_one_shot_iterator()
                next_element = iterator.get_next()
                batch = 0
                cost = 0
                t = time.time()
                start_of_epoch = t
                while True:
                    try:
                        x_batch, y_batch = sess.run(next_element)
                        if (not use_tensorboard) or batch % 100:
                            loss, _ = sess.run([self.model.cost, self.model.train_ops], 
                                feed_dict={self.model.inputs: x_batch, self.model.outputs: y_batch})
                        else:
                            summary, loss, _ = sess.run([merged, self.model.cost, self.model.train_ops], 
                                feed_dict={self.model.inputs: x_batch, self.model.outputs: y_batch})
                            step = (self.epoch_steps * epoch + batch) // 100
                            summary_writer.add_summary(summary, step)
                        cost += np.asscalar(loss)
                        if (batch + 1) % print_interval == 0 and batch > 0:
                            print("Epoch {} avg loss batches {}-{} = {:.4f}".format(epoch + 1, batch - print_interval + 2, batch + 1, cost/500))
                            cost = 0
                        batch += 1
                        if time.time() - t > 3600:
                            # brute force remove all old checkpoints
                            try:
                                shutil.rmtree(self.checkpoint_dir)
                            except FileNotFoundError:
                                pass
                            save_path = saver.save(sess, self.checkpoint_path)
                            print("Model saved in path: %s" % save_path)
                            t = time.time()
                    except tf.errors.OutOfRangeError:
                        break
                print("epoch {} finished at {}".format(
                    epoch + 1,
                    time.strftime("%H:%M:%S", time.gmtime(time.time() - start_of_epoch))
                ))
                shutil.rmtree(self.checkpoint_dir)
                save_path = saver.save(sess, self.checkpoint_path)
                print("Model saved in path: %s" % save_path)
            print("training finished")

    def save_weights(self):
        # condense checkpoint to store infer graph weights only
        self.model.build_graph()
        trainable_variables, _ = self.model.get_variables()
        saver = tf.train.Saver(var_list=trainable_variables)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        with tf.Session() as sess:
            #sess.run(tf.variables_initializer(non_trainable_vars))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.checkpoint_path)
            self.model.initialize_embeddings(sess)
            save_path = saver.save(sess, self.weights_path)
            print("Weights saved in path: %s" % save_path)

    def validate(self):
        dictionary, _ = read_dictionary(self.languages, self.work_dir)
        end_index = get_end_index(dictionary[self.languages[1]])
        self.model.build_graph()
        trainable_variables, _ = self.model.get_variables()
        saver = tf.train.Saver(var_list=trainable_variables)
        test_dataset = self.create_validationset()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.checkpoint_path)
            self.model.initialize_embeddings(sess)
            iterator = test_dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            t = time.time()
            BLEU_SCORE = 0
            # calculate bleu score from 100.000 random sentences from test set
            for batch in range(100000 // self.batch_size):
                try:
                    x_batch, y_batch = sess.run(next_element)
                    y_hat = sess.run(self.model.translation, feed_dict={self.model.inputs: x_batch})
                    score = 0
                    for logit, label in zip(y_hat, y_batch):
                        # remove padding & eol symbol
                        label = [iw for iw in label if iw > 0 and iw != end_index]
                        logit = [iw for iw in logit if iw > 0 and iw != end_index]
                        score += nltk.translate.bleu_score.sentence_bleu([logit], label)
                    score = score * 100 / x_batch.shape[0]
                    BLEU_SCORE += score
                except tf.errors.OutOfRangeError:
                    break
            BLEU_SCORE /= batch
        print("Calculated BLEU score in {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))
        print("BLEU score is {0:.3f}".format(BLEU_SCORE))

    def translate_interactive(self):
        self.model.build_graph()
        index_to_word_map = self.model.index_to_word_map[self.model.languages[1]]
        trainable_variables, _ = self.model.get_variables()

        for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v.name, v.shape)

        saver = tf.train.Saver(var_list=trainable_variables)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, self.checkpoint_path)
            except tf.errors.NotFoundError:
                try:
                    saver.restore(sess, self.weights_path)
                except tf.errors.NotFoundError:
                    raise Exception("Found no training checkpoint, no weights. Train model first.")
            self.model.initialize_embeddings(sess)
            while True:
                sentence = input("Sentence to translate: ")
                if sentence == "exit()":
                    print("bye")
                    break
                inputs = process_line(sentence,
                    self.model.dictionary[self.model.languages[0]], 
                    get_unk_index(self.model.dictionary[self.model.languages[0]]), 
                    self.model.Tx)
                inputs = np.expand_dims(np.array(inputs), axis=0)
                line_as_integers = sess.run(self.model.translation, feed_dict={self.model.inputs: inputs})
                result = indexes_to_words(line_as_integers.tolist()[0], index_to_word_map)
                print("Translation: {}".format(post_process_line(" ".join(result))))

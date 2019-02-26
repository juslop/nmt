from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

def _encoder_cpu_lstm(*args, **kwargs):
    return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(*args, **kwargs)

def _decoder_cpu_lstm(*args, **kwargs):
    return tf.contrib.rnn.LSTMBlockCell(*args, **kwargs)

class Encoder:
    def __init__(self, embeddings_shape, num_layers, units, is_training=False):
        assert num_layers > 1, "at least 2 layers required"
        self.units = units
        self.num_layers = num_layers
        self.is_training = is_training
        self.cells_fw = []
        self.cells_bw = []
        with tf.device("/cpu:0"):
            with tf.variable_scope("encoder_embedding"):
                self.from_embeddings = tf.Variable(tf.constant(0.0, shape=embeddings_shape),
                        trainable=False, name="from_embeddings")
                self.from_embedding_placeholder = tf.placeholder(tf.float32, embeddings_shape,
                    name="from_embeddings_placeholder")
                self.from_embedding_init = self.from_embeddings.assign(self.from_embedding_placeholder)

    def __call__(self, X):
        inputs = self._from_embeddings_lookup(X)
        inputs = tf.transpose(inputs, [1, 0, 2])
        if tf.test.is_gpu_available():
            encoder_outputs = self._cudnnLSTM(inputs)
        else:
            if self.num_layers == 2:
                encoder_outputs = self._bidirectional(inputs)
            else:
                encoder_outputs = self._stacked_bidirectional(inputs)
        return tf.transpose(encoder_outputs, [1, 0, 2])

    def _from_embeddings_lookup(self, X):
        with tf.device("/cpu:0"):
            return tf.nn.embedding_lookup(self.from_embeddings, X)

    def _cudnnLSTM(self, X):
        with tf.variable_scope("encoder"):
            encoder_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(self.num_layers, self.units, direction="bidirectional")
            outputs, _ = encoder_lstm(X, training=self.is_training)
            self.cells_fw = encoder_lstm
            return outputs

    def _bidirectional(self, X):
        assert self.num_layers == 2, "only 2 layers allowed"
        cell_fw = _encoder_cpu_lstm(self.units)
        cell_bw = _encoder_cpu_lstm(self.units)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            X,
            time_major=True,
            dtype=tf.float32,
            scope="encoder"
        )
        self.cells_fw = [cell_fw]
        self.cells_bw = [cell_bw]
        return tf.concat(outputs, -1)

    def _stacked_bidirectional(self, X):
        cells_fw, cells_bw = [], []
        for _ in range(self.num_layers // 2):
            cells_fw.append(_encoder_cpu_lstm(self.units))
            cells_bw.append(_encoder_cpu_lstm(self.units))
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            X,
            dtype = tf.float32,
            time_major=True,
            scope="encoder"
        )
        self.cells_fw = cells_fw
        self.cells_bw = cells_bw
        return outputs

    def log_to_tensorboard(self):
        if tf.test.is_gpu_available():
            #tf.summary.histogram("Encoder", self.cells_fw)
            pass
        else:
            for i, cell in enumerate(self.cells_fw):
                kernel, bias = cell.variables
                tf.summary.histogram("Encoder-{} FW Kernel".format(i), kernel)
                tf.summary.histogram("Encoder-{} FW Bias".format(i), bias)
            for i, cell in enumerate(self.cells_bw):
                kernel, bias = cell.variables
                tf.summary.histogram("Encoder-{} BW Kernel".format(i), kernel)
                tf.summary.histogram("Encoder-{} BW Bias".format(i), bias)

class BaseDecoder(object):
    def __init__(self, Ty, embeddings_shape, to_start_index, end_index, vocabulary_size, num_layers, units, batch_size, is_training=False):
        self.Ty = Ty
        self.num_layers = num_layers
        self.units = units
        self.batch_size = batch_size
        self.to_start_index = to_start_index
        self.vocabulary_size = vocabulary_size
        self.end_index = end_index
        self.is_training = is_training
        with tf.device("/cpu:0"):
            with tf.variable_scope("decoder_embedding"):
                self.to_embeddings = tf.Variable(tf.constant(0.0, shape=embeddings_shape),
                        trainable=False, name="to_embeddings")
                self.to_embedding_placeholder = tf.placeholder(tf.float32, embeddings_shape,
                    name="to_embeddings_placeholder")
                self.to_embedding_init = self.to_embeddings.assign(self.to_embedding_placeholder)

    def _to_embeddings_lookup(self, Y):
        with tf.device("/cpu:0"):
            return tf.nn.embedding_lookup(self.to_embeddings, Y)

    def _decoder_cell(self, encoder_outputs):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units = self.units,
            memory = encoder_outputs)
        cells = [_decoder_cpu_lstm(self.units) for _ in range(self.num_layers)]
        ret = tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell(cells),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.units)
        return ret, cells

class TrainingDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        kwargs["is_training"] = True
        super().__init__(*args, **kwargs)

    def _training_helper(self, Y):
        # feed start index to training helper
        with tf.variable_scope("training_helper_inputs"):
            starts = tf.fill([self.batch_size, 1], self.to_start_index)
            training_Y = tf.slice(Y, [0, 0], [self.batch_size, self.Ty - 1])
            training_Y = tf.concat((starts, training_Y), axis=1)
            helper_inputs = self._to_embeddings_lookup(training_Y)
        return tf.contrib.seq2seq.TrainingHelper(
            inputs = helper_inputs,
            sequence_length = tf.convert_to_tensor([self.Ty]*self.batch_size))

    def __call__(self, encoder_outputs, Y):
        output_layer = tf.layers.Dense(self.vocabulary_size)
        decoder_cells, cells = self._decoder_cell(encoder_outputs)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = decoder_cells,
            helper = self._training_helper(Y),
            initial_state = decoder_cells.zero_state(self.batch_size, tf.float32),
            output_layer = output_layer)
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = training_decoder,
            impute_finished = True,
            maximum_iterations = self.Ty)
        for i, lstm_cell in enumerate(cells):
            kernel, bias = lstm_cell.variables
            tf.summary.histogram("Decoder {} Kernel".format(i), kernel)
            tf.summary.histogram("Decoder {} Bias".format(i), bias)
        for v in output_layer.trainable_variables:
            tf.summary.histogram("Output {}".format(v.name), v)
        return output.rnn_output

class BeamSearchDecoder(BaseDecoder):
    def __init__(self, beam_width, *args, **kwargs):
        self.beam_width = beam_width
        super().__init__(*args, **kwargs)

    def __call__(self, encoder_outputs):
        alpha = 0.7
        beta = 0.4
        output_layer = tf.layers.Dense(self.vocabulary_size)

        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=self.beam_width)

        decoder_cell, _ = self._decoder_cell(tiled_encoder_outputs)
        initial_state=decoder_cell.zero_state(self.batch_size*self.beam_width, dtype=tf.float32)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell = decoder_cell,
            embedding = self._to_embeddings_lookup,
            start_tokens=tf.convert_to_tensor([self.to_start_index] * self.batch_size),
            end_token=self.end_index,
            initial_state=initial_state,
            beam_width=self.beam_width,
            output_layer=output_layer,
            length_penalty_weight=alpha,
            coverage_penalty_weight=beta
        )
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = decoder,
            maximum_iterations = self.Ty)
        #    impute_finished = True,
        return output.predicted_ids[:,:,0]

class GreedyDecoder(BaseDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, encoder_outputs):
        output_layer = tf.layers.Dense(self.vocabulary_size)
        decoder_cells, _ = self._decoder_cell(encoder_outputs)

        decoder = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding = self._to_embeddings_lookup,
            start_tokens=tf.convert_to_tensor([self.to_start_index]*self.batch_size),
            end_token=self.end_index
        )
        infer_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell = decoder_cells,
            helper = decoder,
            initial_state = decoder_cells.zero_state(self.batch_size, tf.float32),
            output_layer = output_layer)
        output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder = infer_decoder,
            impute_finished = True,
            maximum_iterations = self.Ty)
        output = tf.argmax(output.rnn_output, -1, output_type=tf.int32)
        return output

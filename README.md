# nmt

Natural language machine translation experimental supervised deep learning application using Tensorflow 1.13 and Nvidia Cuda 10.0

Ready to train language configurations: english-german-english, english-spanish-english, english-finnish-english.

## Install: 

- Python3 required. 
- install Cuda 10.0 and CUDNN as in tensorflow instructions
- clone git repo
- cd to repo
- create Python virtual env
- activate virtual env
- pip install -r gpu-requirements.txt

## Use:

- copy desired config file from configs dir to config.json file in the translator dir or make your own
- python translate.py -h for instructions
  - main commands: 
    - dictionary: fetch language data and create dictionaries
    - glove: fetch glove vectors and create embedding matrices
    - prepare: convert training files from words to indexes to speed up training
    - train: train language pair, requires above steps, use -r option to swap language order
    - translate: once trained try out infer model
    - validate: calculate BLEU score
   - hyper parameters and neural network architecture can be configured by editing configuration files and with command line parameters

## Try out a trained model in Google Colab (from finnish to english only sorry):

- download weights, embeddings and dictionaries
  - [dictionaries, embeddings](https://drive.google.com/file/d/1SMxGnlWW2YwZmxRSR5cpLkHeZBFxxp__/view?usp=sharing)
  - [weights](https://drive.google.com/file/d/1dkJ7uCQ3qaAxb6CSki1eqKfkhu5B5YT-/view?usp=sharing)
  - Note: Google drive complains about not able to virus scan a large file
- use the ipynb file in colab folder
  - upload the weights and dictionaries to colab or gdrive

## Observations & learnings

- when training, Adam optimizer requires dropping learning rate
  - starting with 0.001 and then going down to 0.0001 after half a million lines of text
- last 10% of loss reduction takes same time as first 90%
- 6 LSTM layers starts to be too much to train without residual connections. However, due to exploding GPU memory requirement, did not try that
  - with 4 layer LSTM model loss went down to 1.58, with 6 layers loss stayed above 1.8
- the size of vocabulary of language to translate to impacts most heavily GPU memory consumption
  - the output dense layers size is vocabulary_size * lstm_units
  - if vocabulary size is 1M and LSTM units 1014, data type float32, that is 4G for model and 2 more of same size for Adam tensors, requires separate GPU with long voc size
- tried this same model with tf2.0 and tf.keras models, but run out of gpu memory immediately
- did the training with a PC with 2 x GeForce GTX 1060 6GB, which is for just playing around. Serious natural machine translation requires: 
  - 8 GPUs of 16GB memory each
  - residual LSTM layer connections
  - longer dictionaries
  - replacing unk with word fractions
  - etc.

## Model weights and their dimensions in finnish to english case

- encoder_embedding/from_embeddings:0 (286472, 300)
- decoder_embedding/to_embeddings:0 (65052, 300)
- encoder/cudnn_lstm/opaque_kernel:0 <unknown>
  - note encoder in gpu memory only
- memory_layer/kernel:0 (1280, 640)
- decoder/attention_wrapper/multi_rnn_cell/cell_0/lstm_cell/kernel:0 (1580, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_0/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/multi_rnn_cell/cell_1/lstm_cell/kernel:0 (1280, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_1/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/multi_rnn_cell/cell_2/lstm_cell/kernel:0 (1280, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_2/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/multi_rnn_cell/cell_3/lstm_cell/kernel:0 (1280, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_3/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/multi_rnn_cell/cell_4/lstm_cell/kernel:0 (1280, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_4/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/multi_rnn_cell/cell_5/lstm_cell/kernel:0 (1280, 2560)
- decoder/attention_wrapper/multi_rnn_cell/cell_5/lstm_cell/bias:0 (2560,)
- decoder/attention_wrapper/bahdanau_attention/query_layer/kernel:0 (640, 640)
- decoder/attention_wrapper/bahdanau_attention/attention_v:0 (640,)
- decoder/attention_wrapper/attention_layer/kernel:0 (1920, 640)
- decoder/dense/kernel:0 (640, 65052)
- decoder/dense/bias:0 (65052,)

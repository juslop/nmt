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

#
#
# data available in e.g.:
# EMNLP 2018 THIRD CONFERENCE ON MACHINE TRANSLATION (WMT18)
# http://statmt.org/wmt18/index.html
# http://statmt.org/wmt18/translation-task.html#download
# Europarl: A Parallel Corpus for Statistical Machine Translation
# http://www.statmt.org/europarl/
# paper:
# http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf
#
#

import os, argparse, json
from prepare import (create_dictionary,
    convert_words_to_indexes, read_dictionary)
from get_data import get_data, fetch_glove_vectors, get_embedding_matrix
from runner import RunNMT

VALIDATION_BATCH_SIZE = 32

def _init_parser():
    parser = argparse.ArgumentParser(
        description="Language translation neural network. python translate.py cmd -h for details for command.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers()

    dictionary_parser = subparsers.add_parser('dictionary', 
    help="""    fetches language files, if not yet in cache\n\
    and creates dictionary for the language pair\n\
    and store the hd5 file to work_dir.
    """)
    dictionary_parser.set_defaults(cmd="dictionary")

    glove_parser = subparsers.add_parser('glove', 
        help="""    fetch glove word vectors for language pair, if not in cache\n\
    and convert to numpy arrays and store to work_dir.
    """)
    glove_parser.set_defaults(cmd="glove")

    prepare_parser = subparsers.add_parser("prepare", 
        help="""    converts the text file pairs to indexes for neural network\n\
    and stores them as numpy files to work_dir.
    """)
    prepare_parser.set_defaults(cmd="prepare")

    prepare_all_parser = subparsers.add_parser("build-all", 
        help="""    does all data preparation: dictionary, glove, prepare.
    """)
    prepare_all_parser.set_defaults(cmd="build_all")

    train_parser = subparsers.add_parser("train", 
        help="""    train the neural network with data from previous steps.
    """)
    train_parser.add_argument("-l", "--load", action="store_true",
        help="continue training from latest checkpoint")
    train_parser.add_argument("-lr", "--learning-rate", type=float,
        help="set learning rate manually")
    train_parser.add_argument("-b", "--batch-size", type=int, 
        help="set batch size. can be larger when translating to english.")
    train_parser.set_defaults(cmd="train")

    weights_parser = subparsers.add_parser("save_weights", 
        help="""    once trained enough, save weights only (no optimizer parameters).
    """)
    weights_parser.set_defaults(cmd="save_weights")

    translate_parser = subparsers.add_parser("translate",
        help="""    translate text.
    """)
    translate_parser.add_argument("-b", "--beam-width", type=int, 
        help="set beam width for beam decoder.")
    translate_parser.set_defaults(cmd="translate")

    validate_parser = subparsers.add_parser("validate",
        help="""    calculate BLEU score.
    """)
    validate_parser.set_defaults(cmd="validate")

    parser.add_argument("-r", "--reverse", action="store_true", 
        help="swap translation order of languages.")

    args = parser.parse_args()
    return args

def main():
    with open('config.json') as json_data_file:
        data = json.load(json_data_file)
    langs = data["languages"]

    args = _init_parser()

    work_dir = data["work_dir"]
    translation_files = data['translation_files']

    # sentence max length, Tx for first language, Ty for the other
    seq_lens = [data["hyper_parameters"]["Tx"], data["hyper_parameters"]["Ty"]]
    batch_size = data["hyper_parameters"]["batch_size"]
    num_layers = data["hyper_parameters"]["num_layers"]
    lstm_units = data["hyper_parameters"]["lstm_units"]
    beam_width = data["hyper_parameters"].get("beam_width", 1)
    test_set_pct = data["hyper_parameters"]["test_set_pct"] # % of all lines
    skip_words_treshold = data["skip_words_treshold"]

    action = args.cmd
    if args.reverse:
        langs.reverse()
        seq_lens.reverse()
    languages = tuple(langs)
    Tx = seq_lens[0]
    Ty = seq_lens[1]

    print("Configured for {} --> {}".format(languages[0], languages[1]))

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if action == 'dictionary':
        file_pairs = get_data(translation_files)
        create_dictionary(languages, file_pairs, skip_words_treshold, work_dir)
    elif action == 'glove':
        dictionaries, _ = read_dictionary(languages, work_dir)
        fetch_glove_vectors(languages, dictionaries, work_dir, data)
    elif action == 'prepare':
        file_pairs = get_data(translation_files)
        convert_words_to_indexes(languages, file_pairs, work_dir, Tx, Ty, test_set_pct)
    elif action == 'build_all':
        # note: tensorflow keras utils do not extract all data types so this may fail, at least so in Windows. 
        # Recommend build in pieces and check every step.
        file_pairs = get_data(translation_files)
        create_dictionary(languages, file_pairs, skip_words_treshold, work_dir)
        dictionaries, _ = read_dictionary(languages, work_dir)
        fetch_glove_vectors(languages, dictionaries, work_dir, data)
        convert_words_to_indexes(languages, file_pairs, work_dir, Tx, Ty, test_set_pct)
    elif action == 'train':
        kwargs={}
        if args.learning_rate:
            kwargs["lr"] = args.learning_rate
        if args.batch_size is not None:
            batch_size = args.batch_size
        instance = RunNMT(langs, Tx, Ty, num_layers,
            lstm_units, batch_size, work_dir, **kwargs)
        kwargs={}
        if args.load:
            kwargs["load_checkpoint"] = True
        instance.train(2, **kwargs)
    elif action == 'save_weights':
        instance = RunNMT(langs, Tx, Ty, num_layers,
            lstm_units, 1, work_dir)
        instance.save_weights()
    elif action == 'translate':
        if args.beam_width is not None:
            beam_width = args.beam_width
        instance = RunNMT(langs, Tx, Ty, num_layers,
            lstm_units, 1, work_dir, beam_width=beam_width)
        instance.translate_interactive()
    elif action == 'validate':
        batch_size = VALIDATION_BATCH_SIZE
        instance = RunNMT(langs, Tx, Ty, num_layers,
            lstm_units, batch_size, work_dir, beam_width=1) # tf beam search is buggy
        instance.validate()
    else:
        print("not implemented yet.")

if __name__ == "__main__":
    main()

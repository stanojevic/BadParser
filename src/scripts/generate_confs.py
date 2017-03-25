#!/usr/bin/env python3

import itertools
from os.path import join, exists
from os import mkdir
from sys import stderr
import argparse


# command for finding the best configuration from log*.err files
# for X in log.train_node_*.err ; do cat $X | grep acc | sed "s/$/  $X/" | sed "s/\(.* acc: \)\([^ ]*\)/\2 \1\2/" ; done | sort -n
def create_all_confs():

    conf_dict = {
        "terminal_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "word_droppiness": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "composition_function_dropout": [0.0, 0.2, 0.4],

        "w_emb_size": [100],  # [30, 100],
        "p_emb_size": [20],
        "n_emb_size": [20],
        "a_emb_size": [20],
        "c_emb_size_for_char": [10],
        "c_emb_size_for_word": [100], # [25, 50, 100]

        "use_pretrained_emb": [1],  # FIXED
        "update_pretrained_emb": [0],  # FIXED

        "bilstm_layers": [2],  # [0,1,2],

        #Tree
        "node_rep_size": [40],   # [30, 40, 50, 60, 70, 80]
        "composition_function": ["TreeLSTM"],

        #Configuration
        "use_configuration_lstm": [0],  # [0, 1]
        "config_rep_size": [100],  # [50, 100]

        #Stack-LSTMs:
        "stack_lstm_layers":   [2],  # [1,2],
        "stack_hidden_size": [100],  # [50, 100],
        "stack_dropout": [0.0],
        "stack_ngram_count": [-1],

        "buffer_lstm_layers":   [2],  # [1,2],
        "buffer_hidden_size": [100],  # [50, 100],
        "buffer_dropout": [0.0],
        "buffer_ngram_count": [-1],

        "action_lstm_layers":   [2],  # [1,2],
        "action_hidden_size": [100],  # [50, 100],
        "action_dropout": [0.0],
        "action_ngram_count": [-1],

        #Fixed:
        "beam_size": [1],
        "laziness": ["laziest"],
        "optimizer": ["Adam"],
        "update_type": ["sparse"]

    }

    confs = conf_dict.values()
    keys = conf_dict.keys()
    all_confs= list(itertools.product(*confs))

    return all_confs, keys


def write_config_files(dirname, prefix):
    if not exists(dirname):
        mkdir(dirname)

    all_confs, keys = create_all_confs()

    properties_to_show_in_filename = \
        ["terminal_dropout", "word_droppiness", "composition_function_dropout"]

    for idx, conf in enumerate(all_confs):
        properties = dict(zip(keys, conf))
        filename = "%s_IDX%d" % (prefix, idx)
        for prop in properties_to_show_in_filename:
            filename += "___"+prop+"_"+str(properties[prop])
        filename+=".json"
        with open(join(dirname, filename), "w") as fh:
            print(join(dirname, filename), file=stderr)
            print("{", file=fh)
            for i, (k, v) in enumerate(zip(keys, conf)):
                line_to_print = "\t"
                if type(v) == str:
                    line_to_print += '"%s": "%s"' % (k, v)
                elif type(v) == int:
                    line_to_print += '"%s": %d' % (k, v)
                else:
                    line_to_print += '"%s": %f' % (k, v)
                if i != len(keys)-1:
                    line_to_print += ","
                print(line_to_print, file=fh)
            print("}", file=fh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--prefix", required=True, type=str)
    args = parser.parse_args()
    write_config_files(args.output_dir, args.prefix)


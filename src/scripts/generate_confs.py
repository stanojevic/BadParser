#!/usr/bin/env python3

import itertools
from os.path import join, exists
from os import mkdir
from sys import stderr
import argparse


# command for finding the best configuration from log*.err files
# for X in log.*.err ; do cat $X | grep acc | sed "s/\(epoch [^ ]*\)\(.* acc: \)\([^ ]*\).*/\3 \1 $X/" ; done | sort -n | less
# for X in log.*.err ; do cat $X | grep acc | sed "s/epoch \([^ ]*\)\(.* acc: \)\([^ ]*\).*/\3 epoch_\1 $X/" ; done | sort -n | sed "s/log\.train_DROPS_IDX[^_]*___//" | sed "s/.err//" | sed "s/___/ /g"
def create_all_confs():

    conf_dict = {
        "terminal_dropout": [0.0], #, 0.1, 0.2, 0.3, 0.4, 0.5], FIXED
        "word_droppiness": [0.0], #, 0.1, 0.2, 0.3, 0.4, 0.5], FIXED
        "tag_droppiness": [0.0], # , 0.1, 0.2, 0.3, 1.0], FIXED
        "composition_function_dropout": [0.0], #, 0.2, 0.4], FIXED

        "w_emb_size": [100],  # [0, 30, 100], FIXED
        "p_emb_size": [20],   # [10, 20] FIXED
        "n_emb_size": [20],   # [10, 20] FIXED
        "a_emb_size": [20],   # [10, 20] FIXED
        "c_emb_size_for_char": [10],  # FIXED
        "c_emb_size_for_word": [100],  # [20, 50, 100] FIXED

        "use_pretrained_emb": [1],  # FIXED
        "update_pretrained_emb": [0], #, 1],  # FIXED

        "bilstm_layers": [2],  # [0,1,2], FIXED

        #Tree
        "node_rep_size": [40],   # [20, 40, 60, 80] FIXED
        "composition_function": ["TreeLSTM"],  # FIXED
        "composition_function_head_ordered": [1], # ,0] FIXED

        #Configuration
        "use_configuration_lstm": [0],  # [0, 1] FIXED
        "config_rep_size": [100],  # [50, 100], FIXED

        #Stack-LSTMs:
        "stack_lstm_layers":  [1],  # [0, 1, 2], FIXED
        "stack_hidden_size": [100],  # [50, 100], FIXED
        "stack_dropout": [0.0],  # FIXED
        "stack_ngram_count": [-1],  # FIXED

        "buffer_lstm_layers":   [1],  # [0, 1, 2], FIXED
        "buffer_hidden_size": [100],  # FIXED
        "buffer_dropout": [0.0],  # FIXED
        "buffer_ngram_count": [-1],  # FIXED

        "action_lstm_layers":  [0], # [0, 1, 2], FIXED
        "action_hidden_size": [0],  # FIXED
        "action_dropout": [0.0],  # FIXED
        "action_ngram_count": [0], # FIXED

        #Fixed:
        "beam_size": [1],  # FIXED
        "laziness": ["eager", "lazy", "lazier"],
        "optimizer": ["Adam"],  # FIXED
        "optimizer_b1": [0.9], # FIXED
        "optimizer_b2": [0.999], # FIXED
        "update_type": ["sparse"],  # FIXED
        "<ind": ["Left", "Right", "RightD", "Dist2", "Label"]
    }

    confs = conf_dict.values()
    keys = conf_dict.keys()
    all_confs= list(itertools.product(*confs))

    ### EXTRA
    for i, key in enumerate(keys):
        if key == "stack_lstm_layers":
            stack_lstm_layers_index = i
        elif key == "buffer_lstm_layers":
            buffer_lstm_layers_index = i
        elif key == "action_lstm_layers":
            action_lstm_layers_index = i
        elif key == "stack_hidden_size":
            stack_hidden_size_index = i
        elif key == "buffer_hidden_size":
            buffer_hidden_size_index = i
        elif key == "action_hidden_size":
            action_hidden_size_index = i
        elif key == "stack_ngram_count":
            stack_ngram_count_index = i
        elif key == "buffer_ngram_count":
            buffer_ngram_count_index = i
        elif key == "action_ngram_count":
            action_ngram_count_index = i
        elif key == "tag_droppiness":
            tag_droppiness_index = i
        elif key == "p_emb_size":
            p_emb_size_index = i

    for i, all_confs_entry in enumerate(all_confs):
        conf = list(all_confs_entry)
        conf[buffer_hidden_size_index] = conf[stack_hidden_size_index]
        conf[action_hidden_size_index] = conf[stack_hidden_size_index]
        if conf[stack_lstm_layers_index] == 0:
            conf[stack_ngram_count_index] = 0
            conf[stack_hidden_size_index] = 0
        if conf[buffer_lstm_layers_index] == 0:
            conf[buffer_ngram_count_index] = 0
            conf[buffer_hidden_size_index] = 0
        if conf[action_lstm_layers_index] == 0:
            conf[action_ngram_count_index] = 0
            conf[action_hidden_size_index] = 0
        if conf[tag_droppiness_index] == 1.0:
            conf[p_emb_size_index] = 2
        all_confs[i] = conf

    properties_to_show_in_filename = []
    for key, value in conf_dict.items():
        if len(value)>1:
            properties_to_show_in_filename.append(key)

    return all_confs, keys, properties_to_show_in_filename


def write_config_files(dirname, prefix):
    if not exists(dirname):
        mkdir(dirname)

    all_confs, keys, properties_to_show_in_filename = create_all_confs()

    #properties_to_show_in_filename = \
    #    ["bilstm_layers", "stack_lstm_layers", "buffer_lstm_layers", "action_lstm_layers", "stack_hidden_size", "config_rep_size"]
    #    #["w_emb_size", "p_emb_size", "n_emb_size", "a_emb_size", "c_emb_size_for_word", "node_rep_size"]
    #    #["update_pretrained_emb"]
    #    #["terminal_dropout", "word_droppiness", "composition_function_dropout"]

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
    print("%d generated"%len(all_confs), file=stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--prefix", required=True, type=str)
    args = parser.parse_args()
    write_config_files(args.output_dir, args.prefix)


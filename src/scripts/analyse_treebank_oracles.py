#!/usr/bin/env python3

import argparse
import json
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr, stdout
import os, sys, inspect
from random import shuffle
import dynet as dy
import re
import numpy as np
from time import time
from math import isinf
from random import random


SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from data_formats.tree_loader import load_from_export_format
from parser.string2int_mapper import String2IntegerMapper, ContainerStr2IntMaps
from parser.configuration import Configuration
from parser.action import ActionStorage
from parser.beam_search import BeamDecoder
from parser.model import *
from data_formats.head_finding import HeadFinder

def _rec_nonterm_mapping(nonterm_set, tree):
    if not tree.is_terminal():
        nonterm_set.add(tree.label)
        for child in tree.children:
            _rec_nonterm_mapping(nonterm_set, child)

def get_nonterm_mapping(trees):
    n2i = String2IntegerMapper()
    nonterm_set = set()
    for tree in trees:
        _rec_nonterm_mapping(nonterm_set, tree)
    for nonterm in sorted(nonterm_set):
        n2i.add_string(nonterm)
    return n2i

def get_pos_mapping(pos_seqs):
    p2i = String2IntegerMapper()
    p2i.add_string(String2IntegerMapper.UNK)
    p_count = dict()
    for pos_seq in pos_seqs:
        for pos in pos_seq:
            p_count[pos] = p_count.get(pos, 0)+1
    for p, _ in sorted(list(filter(lambda x: x[1] >= 3, p_count.items())), key=lambda x: x[0]):
        p2i.add_string(p)
    return p2i

def get_char_mapping(trees):
    MIN_COUNT_KNOWN = 10
    c2i = String2IntegerMapper()
    c2i.add_string(String2IntegerMapper.UNK)
    c_count = dict()
    for tree in trees:
        for node in tree.give_me_terminal_nodes():
            w = node.label
            for c in w:
                c_count[c] = c_count.get(c, 0) + 1
    for c, _ in sorted(list(filter(lambda x: x[1] >= MIN_COUNT_KNOWN, c_count.items())), key=lambda x: x[0]):
        c2i.add_string(c)
    return c2i

def get_word_mapping(trees):
    MIN_COUNT_KNOWN = 3
    w2i = String2IntegerMapper()
    w2i.add_string(String2IntegerMapper.UNK)
    w_count = dict()
    for tree in trees:
        for node in tree.give_me_terminal_nodes():
            w = node.label
            # p = node.attributes["tag"]
            w_count[w] = w_count.get(w, 0)+1
    for w, _ in sorted(list(filter(lambda x: x[1] >= MIN_COUNT_KNOWN, w_count.items())), key=lambda x: x[0]):
        w2i.add_string(w)
    return w2i

def annotate_node_G_ordering(tree, next_free_index, ind_method):
    if ind_method == "Left":
        ordered_children = tree.children
    elif ind_method == "Right":
        ordered_children = reversed(tree.children)
    elif ind_method == "RightD":
        if tree.is_projective :
            ordered_children = tree.children
        else:
            ordered_children = reversed(tree.children)
    elif ind_method == "Dist2":
        if tree.is_gap_creator(2):
            ordered_children = reversed(tree.children)
        else:
            ordered_children = tree.children
    elif ind_method == "Label":
        if tree.label == "NP" or tree.label == "PP" or tree.is_projective :
            ordered_children = tree.children
        else:
            ordered_children = reversed(tree.children)
    else:
        raise Exception("unknown <G method %s"%ind_method)

    for child in ordered_children:
        next_free_index = annotate_node_G_ordering(child, next_free_index, ind_method)
    tree.attributes["<G"] = next_free_index
    return next_free_index+1


def find_me_a_mother(tree, madre="Whatever"):
    tree.attributes['my_mother'] = madre
    for child in tree.children:
        find_me_a_mother(child, tree)

def annotate_projectivity(tree):
    if tree.is_terminal():
        tree.attributes['fully_projective'] = True
        return
    else:
        for child in tree.children:
            annotate_projectivity(child)
        for child in tree.children:
            if child.attributes['fully_projective'] == False:
                tree.attributes['fully_projective'] = False
                return
        words_covered = len(tree.covered_indices)
        if words_covered == (max(tree.covered_indices) - min(tree.covered_indices) + 1):
            tree.attributes['fully_projective'] = True
            return
        else:
            tree.attributes['fully_projective'] = False
            return


def annotate_max_projective_constituent(tree, mpc=-1):
    if mpc == -1 and tree.attributes['fully_projective']:
        tree.attributes['mpc'] = tree.attributes['<G']
        for child in tree.children:
            annotate_max_projective_constituent(child, tree.attributes['<G'])
    else:
        tree.attributes['mpc'] = mpc
        for child in tree.children:
            annotate_max_projective_constituent(child, mpc)

def lazy_satisfied(laziness, conf):
    if laziness != "lazy":
        return True

    if conf.buffer.size == 0 or conf.stack.size == 0:
        return True

    real_stack_top = conf.stack.top().attributes['real_me']
    real_buffer_top = conf.buffer.top().attributes['real_me']

    real_stack_top_mpc = real_stack_top.attributes['mpc']
    real_buffer_top_mpc = real_buffer_top.attributes['mpc']

    if real_stack_top_mpc != -1 and real_buffer_top_mpc != -1 and real_stack_top_mpc == real_buffer_top_mpc:
        return False
    else:
        return True


def annotate_closest_projective_ancestor(tree, closest_proj_anc):
    tree.attributes['cpa'] = closest_proj_anc
    words_covered = len(tree.covered_indices)
    if words_covered == (max(tree.covered_indices) - min(tree.covered_indices) + 1):
        for child in tree.children:
            annotate_closest_projective_ancestor(child, tree.attributes['<G'])
    else:
        for child in tree.children:
            annotate_closest_projective_ancestor(child, closest_proj_anc)


def laziest_satisfied(laziness, conf):
    if laziness != "laziest":
        return True

    if conf.stack.size <= 1:
        return True

    top = conf.stack.top()
    second_top = conf.stack.second_top()
    real_top = top.attributes['real_me']
    real_second_top = second_top.attributes['real_me']

    if not is_complete(top):
        real_top = real_top.children[real_top.attributes['head_child']]
    if not is_complete(second_top):
        real_second_top = real_second_top.children[real_second_top.attributes['head_child']]

    if real_top.attributes['cpa'] == real_second_top.attributes['cpa']:
        return True
    else:
        return False

def is_complete(node):
    return len(node.children) == len(node.attributes['real_me'].children)

def construct_oracle_conf(tree, pos_seq, params, all_s2i, laziness, word_droppiness, tag_droppiness, terminal_dropout_rate, ind_method):
    annotate_node_G_ordering(tree, 0, ind_method)
    find_me_a_mother(tree)
    if laziness == "lazy":
        annotate_projectivity(tree)
        annotate_max_projective_constituent(tree)
    if laziness == "laziest":
        annotate_closest_projective_ancestor(tree, tree.attributes['<G'])

    words = [node.label for node in tree.give_me_terminal_nodes()]
    if word_droppiness > 0:
        for i in range(len(words)):
            rand_num = random()
            if rand_num < word_droppiness:
                words[i] = String2IntegerMapper.DROPPED
    new_pos_seq = pos_seq.copy()
    if tag_droppiness > 0:
        for i in range(len(new_pos_seq)):
            rand_num = random()
            if rand_num < tag_droppiness:
                new_pos_seq[i] = String2IntegerMapper.UNK
    action_storage = ActionStorage(all_s2i.n2i, params['E_a'])
    init_conf = Configuration.construct_init_configuration(words, new_pos_seq, params, action_storage, all_s2i, terminal_dropout_rate)

    leafs = tree.give_me_terminal_nodes()
    buffer_pointer = init_conf.buffer
    while buffer_pointer.size > 0:
        buffer_pointer.top().attributes['real_me'] = leafs[buffer_pointer.top().leftmost_word_position]
        buffer_pointer = buffer_pointer.pop()

    c = init_conf

    while not (c.is_final_configuration() and c.stack.top().label == tree.label):

        # ADJOINS
        # if second_top() is complete &&&&&& top() is its mother
        if c.stack.size >= 2:
            second_top = c.stack.second_top()
            top = c.stack.top()
            # ADJ_LEFT
            if is_complete(second_top) and \
                    second_top.attributes['real_me'].attributes['my_mother'].is_equal_to(top.attributes['real_me']):
                c = c.transition(action_storage.ADJ_LEFT)
                continue
            elif is_complete(top) and \
                    top.attributes['real_me'].attributes['my_mother'].is_equal_to(second_top.attributes['real_me']):
                c = c.transition(action_storage.ADJ_RIGHT)
                continue

                # PRO-X
                # if top() is complete &&&&&& top() is the head of its mother
        if c.stack.size != 0:
            top = c.stack.top()
            real_me = top.attributes['real_me']
            real_mother = real_me.attributes['my_mother']
            mothers_head = real_mother.children[real_mother.attributes['head_child']] if type(real_mother) != str else None
            if mothers_head is not None and \
                    is_complete(top) and \
                    mothers_head.is_equal_to(real_me):
                c = c.transition(action_storage.get_pro_index_for_string_label(real_mother.label))
                c.stack.top().attributes['real_me'] = real_mother
                continue

                # SWAP
                # if second_top()[<G] > top()[<G]
        if c.stack.size >= 2:
            real_me_top = c.stack.top().attributes['real_me']
            if not is_complete(c.stack.top()):
                real_me_top = real_me_top.children[real_me_top.attributes['head_child']]

            real_me_second_top = c.stack.second_top().attributes['real_me']
            if not is_complete(c.stack.second_top()):
                real_me_second_top = real_me_second_top.children[real_me_second_top.attributes['head_child']]

            if real_me_top.attributes['<G'] < real_me_second_top.attributes['<G'] and \
                    lazy_satisfied(laziness, c) and laziest_satisfied(laziness, c):
                c = c.transition(action_storage.SWAP)
                continue

                # SHIFT
                # otherwise
        c = c.transition(action_storage.SHIFT)
        continue

    return c  # -c.log_prob

def count_transitions(final_conf, sent_id):
    all_confs = []
    all_actions = []
    conf = final_conf
    while conf.prev_conf is not None:
        all_confs.append(conf.prev_conf)
        all_actions.append(conf.last_action)
        conf = conf.prev_conf
    all_confs = reversed(all_confs)
    all_actions = reversed(all_actions)

    counts = {}
    swap_block_sizes = []
    swap_alt_block_sizes = []
    const_swap_transition_sizes = set()
    const_swap_block_sizes = []
    const_swap_block_sizes_cummul = 0
    const_swap_consecutive_count = 0
    const_swap_alt_block_sizes = []
    for (conf, action) in zip(all_confs, all_actions):
        if action == "swap":
            counts['swap'] = counts.get("swap", 0) + 1
            const_swap_consecutive_count += 1
            const_swap_block_sizes_cummul += len(conf.stack.second_top().give_me_terminal_nodes())
            swap_block_sizes.append(len(conf.stack.second_top().give_me_terminal_nodes()))
            swap_alt_block_sizes.append(len(conf.stack.top().give_me_terminal_nodes()))
        else:
            if const_swap_consecutive_count > 0:
                counts['comp_swap'] = counts.get("comp_swap", 0) + 1
                const_swap_transition_sizes.add(const_swap_consecutive_count)
                const_swap_consecutive_count = 0
                const_swap_block_sizes.append(const_swap_block_sizes_cummul)
                const_swap_block_sizes_cummul = 0
                const_swap_alt_block_sizes.append(len(conf.stack.top().give_me_terminal_nodes()))


    if 'swap' in counts:
        counts['avg_block_size'] = np.mean(np.array(swap_block_sizes))
        counts['avg_alt_block_size'] = np.mean(np.array(swap_alt_block_sizes))
        counts['comp_avg_block_size'] = np.mean(np.array(const_swap_block_sizes))
        counts['comp_avg_alt_block_size'] = np.mean(np.array(const_swap_alt_block_sizes))
    else:
        counts['swap'] = 0
        counts['avg_block_size'] = 0.0
        counts['avg_alt_block_size'] = 0.0
        counts['comp_swap'] = 0
        counts['comp_avg_block_size'] = 0.0
        counts['comp_avg_alt_block_size'] = 0.0
    counts['comp_transitions'] = const_swap_transition_sizes

    return counts

def count_transitions2(final_conf, sent_id):

    counts = {}
    conf = final_conf
    swap_block_sizes = []
    swap_alt_block_sizes = []
    while conf.prev_conf is not None:
        action = conf.last_action
        if action == "swap":
            counts['swap'] = counts.get("swap", 0) + 1
            swap_block_sizes.append(len(conf.buffer.top().give_me_terminal_nodes()))
            swap_alt_block_sizes.append(len(conf.stack.top().give_me_terminal_nodes()))

        conf = conf.prev_conf


    if 'swap' in counts:
        counts['avg_block_size'] = np.mean(np.array(swap_block_sizes))
        counts['avg_alt_block_size'] = np.mean(np.array(swap_alt_block_sizes))
    else:
        counts['swap'] = 0
        counts['avg_block_size'] = 0.0
        counts['avg_alt_block_size'] = 0.0

    return counts

def main(train_trees_file, train_pos_file, encoding, model_dir, hyper_params_desc_file, external_embeddings_file):
    hyper_params = load_hyper_parameters_from_file(hyper_params_desc_file)

    # load the data in the memory
    train_trees = load_from_export_format(train_trees_file, encoding)
    train_pos_seqs = load_pos_tags(train_pos_file)
    assert(len(train_trees) == len(train_pos_seqs))
    train_data = list(filter(lambda x: x[0] is not None, zip(train_trees,train_pos_seqs)))
    train_trees = list(map(lambda x: x[0], train_data))
    train_pos_seqs = list(map(lambda x: x[1], train_data))
    all_s2i = ContainerStr2IntMaps()
    all_s2i.w2i = get_word_mapping(train_trees)
    if 'c_emb_size_for_char' in hyper_params:
        all_s2i.c2i = get_char_mapping(train_trees)
    else:
        all_s2i.c2i = None
    all_s2i.p2i = get_pos_mapping(train_pos_seqs)
    all_s2i.n2i = get_nonterm_mapping(train_trees)

    hyper_params['w_voc_size'] = all_s2i.w2i.size()
    hyper_params['p_voc_size'] = all_s2i.p2i.size()
    hyper_params['n_voc_size'] = all_s2i.n2i.size()
    hyper_params['a_voc_size'] = all_s2i.n2i.size()+4  # is this correct?
    if all_s2i.c2i is not None:
        hyper_params['c_voc_size'] = all_s2i.c2i.size()
    #laziness = hyper_params['laziness']
    model, params = define_model(hyper_params, all_s2i, external_embeddings_file)

    reporting_frequency = 1000

    train_data = list(zip(train_trees, train_pos_seqs))

    output_fh = stdout

    columns = ["sentID",
               "words",
               "swapsEager",
               "swapsLazy",
               "swapsLazier",
               "avgBlockSizeEager",
               "avgBlockSizeLazy",
               "avgBlockSizeLazier",
               "avgAltBlockSizeEager",
               "avgAltBlockSizeLazy",
               "avgAltBlockSizeLazier",
               "comp_swapsEager",
               "comp_swapsLazy",
               "comp_swapsLazier",
               "comp_avgBlockSizeEager",
               "comp_avgBlockSizeLazy",
               "comp_avgBlockSizeLazier",
               "comp_avgAltBlockSizeEager",
               "comp_avgAltBlockSizeLazy",
               "comp_avgAltBlockSizeLazier",
               "better"]
    print(",".join(columns), file=output_fh)
    # print("sentID,words,swapsEager,swapsLazy,swapsLazier,avgBlockSizeEager,avgBlockSizeLazy,avgBlockSizeLazier,avgAltBlockSizeEager,avgAltBlockSizeLazy,avgAltBlockSizeLazier,better", file=output_fh)
    # add compoundSwapsEager,compoundSwapsLazy,compoundSwapsLazier,compoundAvgBlockSizeEager,compoundAvgBlockSizeLazy,compoundAvgBlockSizeLazier,compoundAvgAltBlockSizeEager,compoundAvgAltBlockSizeLazy,compoundAvgAltBlockSizeLazier

    const_transition_types_count = {}
    for laziness in ["eager", "lazy", "laziest"]:
        const_transition_types_count[laziness] = set()

    for i, (tree, pos_seq) in enumerate(train_data, 1):
        to_out = [str(i), str(len(pos_seq))]
        #out = "%d,%d"%(i, len(pos_seq))
        counts = {}
        for laziness in ["eager", "lazy", "laziest"]:
            dy.renew_cg()
            oracle_conf = construct_oracle_conf(tree, pos_seq, params, all_s2i, laziness, hyper_params['word_droppiness'], hyper_params['tag_droppiness'], hyper_params['terminal_dropout'], hyper_params['<ind'])
            counts[laziness] = count_transitions(oracle_conf, tree.attributes['sent_id'])
            const_transition_types_count[laziness] |= counts[laziness]['comp_transitions']
        # out+=",%d,%d,%d,%f,%f,%f,%f,%f,%f"%(
        #     counts["eager"]['swap'], counts["lazy"]['swap'], counts["laziest"]['swap'],
        #     counts["eager"]['avg_block_size'], counts["lazy"]['avg_block_size'], counts["laziest"]['avg_block_size'],
        #     counts["eager"]['avg_alt_block_size'], counts["lazy"]['avg_alt_block_size'], counts["laziest"]['avg_alt_block_size'])
        to_out.extend(map(str, [
            counts["eager"]['swap'],
            counts["lazy"]['swap'],
            counts["laziest"]['swap'],
            counts["eager"]['avg_block_size'],
            counts["lazy"]['avg_block_size'],
            counts["laziest"]['avg_block_size'],
            counts["eager"]['avg_alt_block_size'],
            counts["lazy"]['avg_alt_block_size'],
            counts["laziest"]['avg_alt_block_size'],
            counts["eager"]['comp_swap'],
            counts["lazy"]['comp_swap'],
            counts["laziest"]['comp_swap'],
            counts["eager"]['comp_avg_block_size'],
            counts["lazy"]['comp_avg_block_size'],
            counts["laziest"]['comp_avg_block_size'],
            counts["eager"]['comp_avg_alt_block_size'],
            counts["lazy"]['comp_avg_alt_block_size'],
            counts["laziest"]['comp_avg_alt_block_size'],
        ]))
        if counts['lazy']['swap']<counts['laziest']['swap']:
            #out+=",yes"
            to_out.append("yes")
        else:
            #out+=",no"
            to_out.append("no")

        print(",".join(to_out), file=output_fh)

        if i % reporting_frequency == 0:
            print("%d"%i, file=stderr)

        stderr.flush()
        output_fh.flush()
    for laziness in ["eager", "lazy", "laziest"]:
        print("const type transitions %s %d"%(laziness,len(const_transition_types_count[laziness])), file=stderr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Model output directory")
    parser.add_argument("--train_trees_file", required=True, type=str, help="export format training file")
    parser.add_argument("--train_pos_file", required=True, type=str, help="stanford tagger format(sep /) training file")
    parser.add_argument("--external_embeddings_file", default=None, type=str, help="csv file with embeddings")
    parser.add_argument("--dynet-mem", default=512, type=int, help="memory for the neural network")
    parser.add_argument("--dynet-weight-decay", default=0, type=float, help="weight decay (L2) for the neural network")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="Export format encoding default=utf-8, alternative latin1")
    parser.add_argument("--hyper_params_file", required=True, type=str, help="file with hyperparameters in json format")
    args = parser.parse_args()

    if not exists(args.train_trees_file):
        raise Exception(args.train_trees_file+" not exists")
    if not exists(args.train_pos_file):
        raise Exception(args.train_pos_file+" not exists")
    if not exists(args.hyper_params_file):
        raise Exception(args.hyper_params_file+" not exists")

    if not exists(args.model_dir):
        mkdir(args.model_dir)

    main(args.train_trees_file, args.train_pos_file, args.encoding, args.model_dir, args.hyper_params_file, args.external_embeddings_file)

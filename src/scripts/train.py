#!/usr/bin/env python3

import argparse
import json
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect
from random import shuffle
import dynet as dy
import re
import numpy as np
from time import time
from math import isinf

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from data_formats.tree_loader import load_from_export_format
from parser.string2int_mapper import String2IntegerMapper
from parser.configuration import Configuration
from parser.action import ActionStorage
from parser.beam_search import BeamDecoder
from parser.model import *
from data_formats.head_finding import HeadFinder


def annotate_node_G_ordering(tree, next_free_index=0):
    for child in tree.children:
        next_free_index = annotate_node_G_ordering(child, next_free_index)
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

def construct_oracle_conf(tree, pos_seq, params, w2i, p2i, n2i, laziness):
    annotate_node_G_ordering(tree)
    find_me_a_mother(tree)
    if laziness == "lazy":
        annotate_projectivity(tree)
        annotate_max_projective_constituent(tree)
    if laziness == "laziest":
        annotate_closest_projective_ancestor(tree, tree.attributes['<G'])

    words = [node.label for node in tree.give_me_terminal_nodes()]
    action_storage = ActionStorage(n2i, params['E_a'])
    init_conf = Configuration.construct_init_configuration(words, pos_seq, params, action_storage, w2i, p2i)

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
            real_me_second_top = c.stack.second_top().attributes['real_me']
            if real_me_top.attributes['<G'] < real_me_second_top.attributes['<G'] and \
               lazy_satisfied(laziness, c) and laziest_satisfied(laziness, c):
                c = c.transition(action_storage.SWAP)
                continue

        # SHIFT
            # otherwise
        c = c.transition(action_storage.SHIFT)
        continue

    return c  # -c.log_prob


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
    p2i.add_string("UNK")
    p_count = dict()
    for pos_seq in pos_seqs:
        for pos in pos_seq:
            p_count[pos] = p_count.get(pos, 0)+1
    for p, _ in sorted(list(filter(lambda x: x[1] >= 3, p_count.items())), key=lambda x: x[0]):
        p2i.add_string(p)
    return p2i

def get_word_mapping(trees):
    w2i = String2IntegerMapper()
    w2i.add_string("UNK")
    w_count = dict()
    for tree in trees:
        for node in tree.give_me_terminal_nodes():
            w = node.label
            # p = node.attributes["tag"]
            w_count[w] = w_count.get(w, 0)+1
    for w, _ in sorted(list(filter(lambda x: x[1] >= 3, w_count.items())), key=lambda x: x[0]):
        w2i.add_string(w)
    return w2i


def save_count_transitions(list_of_count_transitions, file_name):
    transition_names = set()
    for ct in list_of_count_transitions:
        transition_names |= set(ct.keys())
    transition_names = sorted(list(transition_names))

    with open(file_name, "w") as fh:
        print(" ".join(transition_names), file=fh)
        for ct in list_of_count_transitions:
            fields = []
            for t in transition_names:
                fields.append(str(ct.get(t, 0)))
            print(" ".join(fields), file=fh)

def count_transitions(final_conf, sent_id):

    action_counts = {}
    consecutive_pro_count = 0
    conf = final_conf
    action_counts['conseq_pro_3'] = 0
    action_counts['conseq_pro_4'] = 0
    action_counts['conseq_pro_>4'] = 0
    while conf.prev_conf is not None:
        action = conf.last_action
        if action.startswith("pro"):
            consecutive_pro_count += 1
        else:
            consecutive_pro_count = 0
        if consecutive_pro_count == 3:
            action_counts['conseq_pro_3'] = action_counts.get('conseq_pro_3', 0) + 1
        if consecutive_pro_count == 4:
            action_counts['conseq_pro_4'] = action_counts.get('conseq_pro_4', 0) + 1
        if consecutive_pro_count > 4:
            action_counts['conseq_pro_>4'] = action_counts.get('conseq_pro_>4', 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1
        conf = conf.prev_conf

    action_counts['num_words'] = len(final_conf.stack.top().give_me_terminal_nodes())
    action_counts['sent_id'] = sent_id

    return action_counts


def main(train_trees_file, train_pos_file, dev_trees_file, dev_pos_file, encoding, model_dir, epochs, hyper_params_desc_file):
    best_validation_score = 0

    # load the data in the memory
    train_trees = load_from_export_format(train_trees_file, encoding)
    train_pos_seqs = load_pos_tags(train_pos_file)
    dev_trees = load_from_export_format(dev_trees_file, encoding)
    dev_pos_seqs = load_pos_tags(dev_pos_file)
    w2i = get_word_mapping(train_trees)
    p2i = get_pos_mapping(train_pos_seqs)
    n2i = get_nonterm_mapping(train_trees)

    hyper_params = load_hyper_parameters_from_file(hyper_params_desc_file)
    hyper_params['w_voc_size'] = w2i.size()
    hyper_params['p_voc_size'] = p2i.size()
    hyper_params['n_voc_size'] = n2i.size()
    hyper_params['a_voc_size'] = n2i.size()+4 # is this correct?
    laziness = hyper_params['laziness']
    model, params = define_model(hyper_params)

    if hyper_params['optimizer'] == "AdaGrad":
        trainer = dy.AdagradTrainer(model)
    elif hyper_params['optimizer'] == "Adam":
        trainer = dy.AdamTrainer(model)
    elif hyper_params['optimizer'] == "SGD":
        trainer = dy.SimpleSGDTrainer(model)

    reporting_frequency = 1000

    train_data = list(zip(train_trees, train_pos_seqs))
    train_data_size = len(train_data)
    dev_data = list(zip(dev_trees, dev_pos_seqs))
    for epoch in range(1, epochs+1):
        time_epoch_start = time()
        shuffle(train_data)
        closs = 0
        epoch_loss = 0
        train_action_counts=[]
        for i, (tree, pos_seq) in enumerate(train_data, 1):
            dy.renew_cg()
            oracle_conf = construct_oracle_conf(tree, pos_seq, params, w2i, p2i, n2i, laziness)
            train_action_counts.append(count_transitions(oracle_conf, tree.attributes['sent_id']))
            loss = -oracle_conf.log_prob
            loss_value = loss.value()
            if isinf(loss_value):
                print("INF LOSS ON: %s"%(tree.__repr__()), file=stderr)
            epoch_loss += loss_value
            closs += loss.value()
            if i % reporting_frequency == 0:
                closs /= reporting_frequency
                print(file=stderr)
                print("epoch %d trees %d/%d closs %f on last %d"%(epoch, i, train_data_size, closs, reporting_frequency), file=stderr)
                closs = 0
                time_passed = time() - time_epoch_start
                trees_per_second = i/time_passed
                minutes_for_epoch = int(train_data_size/(60*trees_per_second))
                print("%dh %dm in total for epoch"%(int(minutes_for_epoch/60), minutes_for_epoch%60), file=stderr)
                minutes_for_epoch_left = int((train_data_size-i)/(60*trees_per_second))
                print("%dh %dm left for epoch"%(int(minutes_for_epoch_left/60), minutes_for_epoch_left%60), file=stderr)

            print(".", end="", file=stderr)
            stderr.flush()

            loss.backward()
            trainer.update()
        trainer.update_epoch()

        validation_score, validation_score30, validation_score40 =\
            validate(epoch, model_dir, params, w2i, p2i, n2i, hyper_params['beam_size'], dev_data)
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            save_model(model, params, hyper_params, model_dir, w2i, p2i, n2i)
        epoch_loss /= train_data_size
        print("epoch %d acc: %f acc30: %f acc40: %f loss: %f"%(epoch, validation_score, validation_score30, validation_score40, epoch_loss), file=stderr)


        if epoch == 1:
            save_count_transitions(train_action_counts, join(model_dir, "transition_counts_training_set_log.csv"))
        nswaps = np.array([d.get('swap', 0) for d in train_action_counts])
        mean_swaps = nswaps.mean()
        std_swaps = nswaps.std()
        print("epoch %d TRAIN mean swaps: %f std swap: %f"%(epoch, mean_swaps, std_swaps), file=stderr)


def validate(epoch, model_dir, params, w2i, p2i, n2i, beam_size, dev_data):
    beam = BeamDecoder(params, w2i, p2i, n2i, beam_size)
    acc = 0
    acc30 = 0
    acc40 = 0
    ntrees = 0
    ntrees30 = 0
    ntrees40 = 0

    dev_data_size = len(dev_data)
    time_start = time()
    word_count_total = 0
    if not exists(join(model_dir, "parse_time_log")):
        mkdir(join(model_dir, "parse_time_log"))
    if not exists(join(model_dir, "transition_counts_validation_set_log")):
        mkdir(join(model_dir, "transition_counts_validation_set_log"))
    fh = open(join(model_dir, "parse_time_log", "parse_time_%d"%epoch), "w")
    print("length seconds", file=fh)
    dev_action_counts=[]
    for i, (tree, pos_seq) in enumerate(dev_data, 1):
        time_sent_start = time()
        words = [node.label for node in tree.give_me_terminal_nodes()]
        predicted_conf, predicted_tree = beam.decode(words, pos_seq)
        escore = eval_score(predicted_tree, tree)
        acc += escore
        ntrees += 1
        #Count actions
        dev_action_counts.append(count_transitions(predicted_conf, i))


        # Cutoffs
        if len(words) <= 30:
            acc30 += escore
            ntrees30 += 1

        if len(words) <= 40:
            acc40 += escore
            ntrees40 += 1

        time_sent_end = time()
        print("%d %f" % (len(words), time_sent_end-time_sent_start), file=fh)

        word_count_total += len(words)

        if i % 1000 == 0:
            print(file=stderr)
            print("validation %d/%d"%(i, dev_data_size), file=stderr)
        print(".", end="", file=stderr)
        stderr.flush()
    fh.close()
    time_end = time()
    period = time_end-time_start
    sents_per_second = dev_data_size/period
    words_per_second = word_count_total/period
    print(file=stderr)
    print("epoch %d sents/sec: %f words/sec: %f"%(epoch, sents_per_second, words_per_second), file=stderr)
    save_count_transitions(dev_action_counts, join(model_dir, "transition_counts_validation_set_log", "epoch_%d"%epoch))

    #Mean action counts
    nswaps = np.array([d.get('swap', 0) for d in dev_action_counts])
    mean_swaps = nswaps.mean()
    std_swaps = nswaps.std()
    print("epoch %d DEV mean swaps: %f std swap: %f"%(epoch, mean_swaps, std_swaps), file=stderr)

    return acc/ntrees, acc30/ntrees30, acc40/ntrees40

def eval_score(predicted_tree, gold_tree):
    gold_labelled_brackettings = dict()
    for node in gold_tree.give_me_nonterminal_nodes():
        representation = node.label + ",".join(map(str, sorted(list(node.covered_indices))))
        gold_labelled_brackettings[representation] = gold_labelled_brackettings.get(representation, 0)+1
    predicted_labelled_brackettings = dict()
    for node in predicted_tree.give_me_nonterminal_nodes():
        representation = node.label + ",".join(map(str, sorted(list(node.covered_indices))))
        predicted_labelled_brackettings[representation] = predicted_labelled_brackettings.get(representation, 0)+1

    matched = 0
    for label, gold_count in gold_labelled_brackettings.items():
        matched += min(gold_count, predicted_labelled_brackettings.get(label, 0))

    p = matched/len(predicted_labelled_brackettings)
    r = matched/len(gold_labelled_brackettings)
    if p+r == 0:
        f = 0
    else:
        f = 2*p*r/(p+r)
    return f


# BASELINE_DIR=models/baseline_model
# mkdir -p $BASELINE_DIR
# cp src/model_templates/hyper_parameters.json
# sbatch -J train -e log.train.err -o log.train.std --wrap="python3 src/scripts/train.py --model_dir models/baseline --train_trees_file data_preprocessed/tiger/tiger_hn_split/train.export --train_pos_file data_preprocessed/tiger/tiger_hn_split/train.tags.gold --dev_trees_file data_preprocessed/tiger/tiger_hn_split/dev.export --dev_pos_file data_preprocessed/tiger/tiger_hn_split/dev.tags.gold --dynet-mem 4000 --epochs 20 --hyper_params_file src/model_templates/hyper_parameters.json"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Model output directory")
    parser.add_argument("--train_trees_file", required=True, type=str, help="export format training file")
    parser.add_argument("--train_pos_file", required=True, type=str, help="stanford tagger format(sep /) training file")
    parser.add_argument("--dev_trees_file", required=True, type=str, help="export format development file")
    parser.add_argument("--dev_pos_file", required=True, type=str, help="stanford tagger format(sep /) development file")
    parser.add_argument("--dynet-mem", default=512, type=int, help="memory for the neural network")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="Export format encoding default=utf-8, alternative latin1")
    parser.add_argument("--epochs", required=True, type=int, help="number of epochs")
    parser.add_argument("--hyper_params_file", required=True, type=str, help="file with hyperparameters in json format")
    args = parser.parse_args()

    if not exists(args.train_trees_file):
        raise Exception(args.train_trees_file+" not exists")
    if not exists(args.train_pos_file):
        raise Exception(args.train_pos_file+" not exists")
    if not exists(args.dev_trees_file):
        raise Exception(args.dev_trees_file+" not exists")
    if not exists(args.dev_pos_file):
        raise Exception(args.dev_pos_file+" not exists")
    if not exists(args.hyper_params_file):
        raise Exception(args.hyper_params_file+" not exists")

    if not exists(args.model_dir):
        mkdir(args.model_dir)

    main(args.train_trees_file, args.train_pos_file, args.dev_trees_file, args.dev_pos_file, args.encoding, args.model_dir, args.epochs, args.hyper_params_file)


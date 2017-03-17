#!/usr/bin/env python3

import argparse
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect
from random import shuffle
import dynet as dy
import re
import json
from time import time

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from parser.model import *
from parser.beam_search import BeamDecoder


def main(model_dir, test_sentences_file, test_pos_file, out_file, beam_size):
    model, params, hyper_params, w2i, p2i, n2i = load_model(model_dir)
    if beam_size:
        hyper_params['beam_size'] = beam_size

    beam = BeamDecoder(params, w2i, p2i, n2i, hyper_params['beam_size'])

    pos_tags = load_pos_tags(test_pos_file)
    sentences = load_sentences(test_sentences_file)
    test_data = zip(sentences, pos_tags)

    word_count = 0
    time_started = time()

    with open(out_file, "w") as fh:
        processed = 0
        for word_seq, pos_seq in test_data:
            word_count += len(word_seq)
            predicted_conf, predicted_tree = beam.decode(word_seq, pos_seq)
            processed += 1
            print(predicted_tree.to_export(processed), file=fh)
            if processed % 5 == 0:
                print("processed %d"%processed, file=stderr)
    time_ended = time()
    sents_per_sec = len(sentences)/(time_ended-time_started)
    words_per_sec = word_count/(time_ended-time_started)
    print("sents/sec: %f words/sec: %f"%(sents_per_sec, words_per_sec), file=stderr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Model output directory")
    parser.add_argument("--test_sentences_file", required=True, type=str, help="sentences to parse")
    parser.add_argument("--test_pos_file", required=True, type=str, help="stanford tagger format(sep /) test file")
    parser.add_argument("--output_file", required=True, type=str, help="file to output the trees in")
    parser.add_argument("--dynet-mem", default=512, type=int, help="memory for the neural network")
    parser.add_argument("--beam_size", type=int, help="beam size")
    args = parser.parse_args()

    if not exists(args.test_sentences_file):
        raise Exception(args.test_sentences_file+" not exists")
    if not exists(args.test_pos_file):
        raise Exception(args.test_pos_file+" not exists")
    if not exists(args.model_dir):
        raise Exception(args.model_dir+" not exists")

    main(args.model_dir, args.test_sentences_file, args.test_pos_file, args.output_file, args.beam_size)

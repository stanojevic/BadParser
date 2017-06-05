#!/usr/bin/env python3

import argparse
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect
from math import floor

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from data_formats.tree_loader import load_from_export_format
from scripts.tiger import save_trees, save_sents, save_tags


def negra_split(trees):
    total_size = len(trees)
    train_size = floor(0.8*total_size)
    dev_size = floor(0.1*total_size)
    test_size = floor(0.1*total_size)
    diff = total_size-(train_size+dev_size+test_size)
    train_size += diff

    train_trees = trees[0:train_size]
    dev_trees = trees[train_size:train_size+dev_size]
    test_trees = trees[train_size+dev_size:]

    return train_trees, dev_trees, test_trees

def main(negra_file, encoding, output_folder):
    trees = load_from_export_format(negra_file, encoding)
    filtered_trees = []
    for tree in trees:
        sent_length = len(tree.give_me_terminal_nodes())
        if sent_length <= 30:
            filtered_trees.append(tree)

    if not exists(output_folder):
        mkdir(output_folder)


    train, dev, test = negra_split(filtered_trees)
    curr_output_folder = join(output_folder, "negra_30")
    if not exists(curr_output_folder):
        mkdir(curr_output_folder)

    save_trees(train, join(curr_output_folder, "train.export"))
    save_trees(dev, join(curr_output_folder, "dev.export"))
    save_trees(test, join(curr_output_folder, "test.export"))

    save_tags(train, join(curr_output_folder, "train.tags.gold"))
    save_tags(dev, join(curr_output_folder, "dev.tags.gold"))
    save_tags(test, join(curr_output_folder, "test.tags.gold"))

    save_sents(train, join(curr_output_folder, "train.raw"))
    save_sents(dev, join(curr_output_folder, "dev.raw"))
    save_sents(test, join(curr_output_folder, "test.raw"))





    train, dev, test = negra_split(trees)
    curr_output_folder = join(output_folder, "negra_all")
    if not exists(curr_output_folder):
        mkdir(curr_output_folder)

    save_trees(train, join(curr_output_folder, "train.export"))
    save_trees(dev, join(curr_output_folder, "dev.export"))
    save_trees(test, join(curr_output_folder, "test.export"))

    save_tags(train, join(curr_output_folder, "train.tags.gold"))
    save_tags(dev, join(curr_output_folder, "dev.tags.gold"))
    save_tags(test, join(curr_output_folder, "test.tags.gold"))

    save_sents(train, join(curr_output_folder, "train.raw"))
    save_sents(dev, join(curr_output_folder, "dev.raw"))
    save_sents(test, join(curr_output_folder, "test.raw"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", required=True, type=str, help="Output folder")
    parser.add_argument("--negra_file", required=True, type=str, help="Negra location unzipped")
    parser.add_argument("--encoding", type=str, default="utf-8", # latin1
                        help="Export format encoding default=utf-8, alternative latin1")
    args = parser.parse_args()

    if not exists(args.negra_file):
        raise Exception(args.negra_file+" not exists")

    main(args.negra_file, args.encoding, args.output_folder)

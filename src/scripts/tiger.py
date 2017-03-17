#!/usr/bin/env python3

import argparse
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

from data_formats.tree_loader import load_from_export_format



#NOTE1 requirement call treetools for scripts and outputing export format
# treetools transform tiger_release_aug07.corrected.16012013.xml tiger.continuous.export --trans root_attach --src-format tigerxml --dest-format export

#NOTE2 input might be ether in latin1 or utf-8 encoding

def standard_split(trees):
    train = trees[0:-10000]
    dev = trees[-10000:-5000]
    test = trees[-5000:]
    return train, dev, test


def hall_and_nivre_split(trees):
    train = list(filter(lambda t: t.attributes['sent_id'] % 10 > 1, trees))
    dev = list(filter(lambda t: t.attributes['sent_id'] % 10 == 1, trees))
    test = list(filter(lambda t: t.attributes['sent_id'] % 10 == 0, trees))
    return train, dev, test


def save_trees(trees, filename):
    with open(filename, "w") as fh:
        for i, tree in enumerate(trees):
            representation = tree.to_export(i+1)
            print(representation, file=fh)


def save_tags(trees, filename):
    with open(filename, "w") as fh:
        for tree in trees:
            tags = tree.to_tags()
            words = tree.to_sentence()
            representation = " ".join(list(map(lambda x: x[0]+"_"+x[1], zip(words, tags))))
            print(representation, file=fh)


def save_sents(trees, filename):
    with open(filename, "w") as fh:
        for tree in trees:
            representation = " ".join(tree.to_sentence())
            print(representation, file=fh)


def main(tiger_file, encoding, output_folder):

    trees = load_from_export_format(tiger_file, encoding)
    filtered_trees = []
    for tree in trees:
        if tree.attributes['sent_id'] not in [46234, 50224]:
            print("filtered sentence %d"%tree.attributes['sent_id'], file=stderr)
            filtered_trees.append(tree)

    std_train, std_dev, std_test = standard_split(filtered_trees)
    hn_train, hn_dev, hn_test = hall_and_nivre_split(filtered_trees)

    if not exists(output_folder):
        mkdir(output_folder)
    tiger_std_dir = join(output_folder, "tiger_std_split")
    tiger_hn_dir = join(output_folder, "tiger_hn_split")
    if not exists(tiger_std_dir):
        mkdir(tiger_std_dir)
    if not exists(tiger_hn_dir):
        mkdir(tiger_hn_dir)

    print("TIGER standard", file=stderr)
    print("saving trees", file=stderr)
    save_trees(std_train, join(tiger_std_dir, "train.export"))
    save_trees(std_dev, join(tiger_std_dir, "dev.export"))
    save_trees(std_test, join(tiger_std_dir, "test.export"))

    print("saving tags", file=stderr)
    save_tags(std_train, join(tiger_std_dir, "train.tags.gold"))
    save_tags(std_dev, join(tiger_std_dir, "dev.tags.gold"))
    save_tags(std_test, join(tiger_std_dir, "test.tags.gold"))

    print("saving words", file=stderr)
    save_sents(std_train, join(tiger_std_dir, "train.raw"))
    save_sents(std_dev, join(tiger_std_dir, "dev.raw"))
    save_sents(std_test, join(tiger_std_dir, "test.raw"))

    print("TIGER H&N", file=stderr)
    print("saving trees", file=stderr)
    save_trees(hn_train, join(tiger_hn_dir, "train.export"))
    save_trees(hn_dev, join(tiger_hn_dir, "dev.export"))
    save_trees(hn_test, join(tiger_hn_dir, "test.export"))

    print("saving tags", file=stderr)
    save_tags(hn_train, join(tiger_hn_dir, "train.tags.gold"))
    save_tags(hn_dev, join(tiger_hn_dir, "dev.tags.gold"))
    save_tags(hn_test, join(tiger_hn_dir, "test.tags.gold"))

    print("saving words", file=stderr)
    save_sents(hn_train, join(tiger_hn_dir, "train.raw"))
    save_sents(hn_dev, join(tiger_hn_dir, "dev.raw"))
    save_sents(hn_test, join(tiger_hn_dir, "test.raw"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", required=True, type=str, help="Output folder")
    parser.add_argument("--tiger_file", required=True, type=str, help="Tiger location unzipped")
    parser.add_argument("--encoding", type=str, default="utf-8", # used to be latin1
                        help="Export format encoding default=utf-8, alternative latin1")
    args = parser.parse_args()

    if not exists(args.tiger_file):
        raise Exception(args.tiger_file+" not exists")

    main(args.tiger_file, args.encoding, args.output_folder)


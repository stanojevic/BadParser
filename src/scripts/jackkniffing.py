#!/usr/bin/env python3

import argparse
from os import listdir, mkdir, pardir, system
from os.path import join, exists, dirname, basename
from sys import stderr
import os, sys, inspect
from math import ceil
import re

SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
MAIN_FOLDER = join(SCRIPT_FOLDER, pardir, pardir)
MODULES_FOLDER = join(MAIN_FOLDER, "src")
if MODULES_FOLDER not in sys.path:
    sys.path.insert(0, MODULES_FOLDER)

architecture = "left3words"  # "bidirectional5words"
separator = "_"

# rm data_preprocessed/negra/*/*.tags.predicted
# sbatch -J cnegraJack -e log.cnegraJack.err -o log.cnegraJack.std --wrap="python3 src/scripts/jackkniffing.py --stanford_tagger_dir stanford-postagger-2016-10-31 --folds 10 --train_tag_file data_preprocessed/negra/continuous/train.tags.gold --dev_tag_file data_preprocessed/negra/continuous/dev.tags.gold --test_tag_file data_preprocessed/negra/continuous/test.tags.gold"
# sbatch -J dnegraJack -e log.dnegraJack.err -o log.dnegraJack.std --wrap="python3 src/scripts/jackkniffing.py --stanford_tagger_dir stanford-postagger-2016-10-31 --folds 10 --train_tag_file data_preprocessed/negra/discontinuous/train.tags.gold --dev_tag_file data_preprocessed/negra/discontinuous/dev.tags.gold --test_tag_file data_preprocessed/negra/discontinuous/test.tags.gold"
# rm data_preprocessed/tiger/*/*.tags.predicted
# sbatch -J stigerJack -e log.stigerJack.err -o log.stigerJack.std --wrap="python3 src/scripts/jackkniffing.py --stanford_tagger_dir stanford-postagger-2016-10-31 --folds 10 --train_tag_file data_preprocessed/tiger/tiger_std_split/train.tags.gold --dev_tag_file data_preprocessed/tiger/tiger_std_split/dev.tags.gold --test_tag_file data_preprocessed/tiger/tiger_std_split/test.tags.gold"
# sbatch -J hntigerJack -e log.hntigerJack.err -o log.hntigerJack.std --wrap="python3 src/scripts/jackkniffing.py --stanford_tagger_dir stanford-postagger-2016-10-31 --folds 10 --train_tag_file data_preprocessed/tiger/tiger_hn_split/train.tags.gold --dev_tag_file data_preprocessed/tiger/tiger_hn_split/dev.tags.gold --test_tag_file data_preprocessed/tiger/tiger_hn_split/test.tags.gold"



def load_lines(fn):
    lines = []
    with open(fn) as fh:
        for line in fh:
            lines.append(line.rstrip())
    return lines

def split_into_chunks(train_lines, folds):
    chunks = []
    chunk_size = ceil(len(train_lines)/folds)
    for current_position in range(0, len(train_lines), chunk_size):
        chunks.append(train_lines[current_position:current_position+chunk_size])
    return chunks

def save_tagged_lines(tagged_lines, output_file):
    with open(output_file, "w") as fh:
        for line in tagged_lines:
            print(line, file=fh)

def save_tagged_lines_without_tags(tagged_lines, output_file):
    with open(output_file, "w") as fh:
        for tagged_line in tagged_lines:
            line = re.sub("%s[^%s ]+ "%(separator, separator), " ", tagged_line)
            line = re.sub("%s[^%s ]+$"%(separator, separator), "", line)
            print(line, file=fh)


def train_and_tag_stanford(stanford_tagger_dir, fold_train, fold_test):
    tmp_dir = "tmp"
    if not exists(tmp_dir):
        mkdir(tmp_dir)

    train_fn = join(tmp_dir, "train")
    save_tagged_lines(fold_train, train_fn)
    test_fn = join(tmp_dir, "test")
    save_tagged_lines_without_tags(fold_test, test_fn)

    model_fn = join(tmp_dir, "model") # bidirectional5words
    train_cmd = "java -classpath %s/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -arch %s -tagSeparator %s -nthreads 2  -sentenceDelimiter newline -tokenize false -model %s -trainFile %s"%(stanford_tagger_dir, architecture, separator, model_fn, train_fn)
    system(train_cmd)

    test_out_fn = join(tmp_dir, "test-tagged")
    test_cmd = "java -mx1300m -classpath %s/stanford-postagger.jar edu.stanford.nlp.tagger.maxent.MaxentTagger -preserveLines -sentenceDelimiter newline -tokenize false  -model %s -textFile %s > %s"%(stanford_tagger_dir, model_fn, test_fn, test_out_fn)
    system(test_cmd)

    return load_lines(test_out_fn)

def transform_fn_to_predicted(gold_file_name):
    base = basename(gold_file_name)
    dir = dirname(gold_file_name)
    return join(dir, base.replace("gold", "predicted"))

def main(stanford_tagger_dir, folds, train_tag_file, dev_tag_file, test_tag_file):
    train_lines = load_lines(train_tag_file)
    chunks = split_into_chunks(train_lines, folds)

    train_tagged_lines = []
    for fold in range(folds):
        fold_test = chunks[fold]
        fold_train = []
        for i, chunk in enumerate(chunks):
            if i != fold:
                fold_train.extend(chunk)
        tagged_lines = train_and_tag_stanford(stanford_tagger_dir, fold_train, fold_test)
        train_tagged_lines.extend(tagged_lines)
    save_tagged_lines(train_tagged_lines, transform_fn_to_predicted(train_tag_file))

    test_lines = load_lines(test_tag_file)
    tagged_test_lines = train_and_tag_stanford(stanford_tagger_dir, train_lines, test_lines)
    save_tagged_lines(tagged_test_lines, transform_fn_to_predicted(test_tag_file))

    dev_lines = load_lines(dev_tag_file)
    tagged_dev_lines = train_and_tag_stanford(stanford_tagger_dir, train_lines, dev_lines)
    save_tagged_lines(tagged_dev_lines, transform_fn_to_predicted(dev_tag_file))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stanford_tagger_dir", required=True, type=str, help="Stanford tagger dir")
    parser.add_argument("--folds", default=10, type=int, help="folds")
    parser.add_argument("--train_tag_file", required=True, type=str, help="train gold tag file")
    parser.add_argument("--dev_tag_file", required=True, type=str, help="train gold tag file")
    parser.add_argument("--test_tag_file", required=True, type=str, help="train gold tag file")
    args = parser.parse_args()

    main(args.stanford_tagger_dir, args.folds, args.train_tag_file, args.dev_tag_file, args.test_tag_file)


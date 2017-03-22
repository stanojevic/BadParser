#!/usr/bin/env python2

import pickle
from os import listdir, mkdir, pardir
from os.path import join, exists
import argparse
import sys
import codecs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--polyglot_pickle_file", required=True, type=str)
    parser.add_argument("--output_txt_file", required=True, type=str)
    parser = parser.parse_args()

    if not exists(parser.polyglot_pickle_file):
        print("pickle file doesn't exist")
        sys.exit(-1)

    in_fh = open(parser.polyglot_pickle_file, "rb")
    out_fh = codecs.open(parser.output_txt_file, "w", encoding="utf8")

    words, embeddings = pickle.load(in_fh)

    for i in range(len(words)):
        repr = words[i]
        repr += "\t"
        repr += "\t".join(map(str, embeddings[i].tolist()))
        repr += "\n"
        out_fh.write(repr)

    in_fh.close()
    out_fh.close()

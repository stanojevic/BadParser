#!/usr/bin/env python3

import argparse
import json
from os import listdir, mkdir, pardir
from os.path import join, exists, basename
from sys import stderr
import os, sys, inspect
from random import shuffle
import re
import numpy as np
from time import time
from math import isinf
from random import random
from glob import glob


def get_res_from_discodop(file):
    res = {}
    with open(file) as fh:
        for line in fh:
            if line.startswith("labeled") or line.startswith("exact"):
                fields = re.split("\\s+", line.rstrip())
                if fields[1].startswith("precision"):
                    res["P"] = fields[3]
                elif fields[1].startswith("recall"):
                    res["R"] = fields[3]
                elif fields[1].startswith("f-measure"):
                    res["F"] = fields[3]
                elif fields[0].startswith("exact"):
                    res["E"] = fields[3]
                else:
                    raise Exception("wtf")
    return res


def get_sys2results_mapping(dir_with_discodop_files):
    disco_res = glob("%s/*.disco"%dir_with_discodop_files)

    sys_res_map = {}

    for res in disco_res:
        sys_name = basename(res).replace(".disco", "").replace("_", "A").replace("0","zero").replace("1", "one").replace("2", "two").replace("3", "three")
        sys_res_map[sys_name] = {}
        disco_res_values = get_res_from_discodop(res)
        for key, val in disco_res_values.items():
            sys_res_map[sys_name]["d"+key] = val
        all_res_values = get_res_from_discodop(res.replace(".disco", ".all"))
        for key, val in all_res_values.items():
            sys_res_map[sys_name]["a"+key] = val

    return sys_res_map

def main(dir_with_discodop_files):
    COLUMNS_TO_PRINT = ["system", 'aP', 'aR', 'aF', 'aE', 'dP', 'dR', 'dF', 'dE']
    FINE_NAME_TO_PRINT = ["system", 'aP', 'aR', 'aF', 'aE', 'dP', 'dR', 'dF', 'dE']
    all_scores =  get_sys2results_mapping(dir_with_discodop_files)
    for sys, scores in all_scores.items():
        print("\\newcommand{\\%s}{%s}"%(sys, sys))
    print("\\begin{table}")
    print("\\begin{tabular}{"+"l|"+"l"*(len(COLUMNS_TO_PRINT)-1)+"}")
    print(" & ".join(FINE_NAME_TO_PRINT)+" \\\\\\hline")
    for sys, scores in all_scores.items():
        scores['system'] = "\\%s"%sys
        print(" & ".join(map(lambda c: scores[c], COLUMNS_TO_PRINT)) +  "\\\\")

    print("\\end{tabular}")
    print("\\caption{Rezultati}")
    print("\\end{table}")

    print("Hello")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_with_discodop_files", required=True, type=str, help="Model output directory")
    args = parser.parse_args()

    if not exists(args.dir_with_discodop_files):
        raise Exception(args.dir_with_discodop_files+" not exists")

    main(args.dir_with_discodop_files)

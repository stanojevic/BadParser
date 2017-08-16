
import re
import codecs
from sys import stderr
from data_formats.tree_representation import TreeNode

from data_formats.head_finding import HeadFinder

import os, sys, inspect
from os.path import join, exists


def load_from_export_format(export_file, encoding):

    trees = []
    SCRIPT_FOLDER = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    hf = HeadFinder(join(SCRIPT_FOLDER, "negra.headrules"))

    with codecs.open(export_file, encoding=encoding) as fh:
        sent_id = None
        buffered_lines = []
        for line in fh:
            if line.startswith("#BOS"):
                sent_id = int(line.split(" ")[1])
            elif line.startswith("#EOS"):
                sent_id2 = int(line.split(" ")[1])
                assert(sent_id == sent_id2)
                if len(buffered_lines) > 0:
                    tree = _give_me_a_tree_from_export_format(buffered_lines)
                    tree.attributes["sent_id"] = sent_id
                    hf.mark_head(tree)
                    trees.append(tree)
                else:
                    trees.append(None)
                if sent_id % 1000 == 0:
                    print("loaded %d trees" % sent_id, file=stderr)
                    stderr.flush()
                sent_id = None
                buffered_lines = []
            elif sent_id is not None:
                buffered_lines.append(line)
            else:
                raise Exception("oh nooooooooo")

    return trees


def _give_me_a_tree_from_export_format(lines):

    word_position = 0
    children = dict()  # children = { parent_id: [child1, child2, ...] }

    for line in lines:

        new_line = re.sub("%%.*", "", line)
        fields = re.split("[\t ]+", new_line.rstrip().lstrip("#"))
        if re.match("#\d+\t", line):  # Constituent
            if len(fields) < 5:
                raise Exception("problem "+line)
            my_id = int(fields[0])
            my_label = fields[1]  # "tag"
            my_morph_tag = fields[2]  # not used
            my_edge = fields[3]  # not used
            parent_id = int(fields[4])

            if my_id == 0:
                raise Exception("root node should not have an entry in the treebank")

            node = TreeNode(my_label, children[my_id], {"morph_tag": my_morph_tag, "edge": my_edge})
        else:  # Word
            my_word = fields[0]
            my_tag = fields[1]
            my_morph_tag = fields[2]  # not used
            my_edge = fields[3]  # not used
            parent_id = int(fields[4])

            node = TreeNode(my_word, [], {
                "tag": my_tag,
                "morph_tag": my_morph_tag,
                "edge": my_edge,
                "word_position": word_position
                },
                word_position=word_position)
            word_position += 1

        if parent_id not in children:
            children[parent_id] = []
        children[parent_id].append(node)


    root_node = TreeNode("root", children[0], {})

    return root_node



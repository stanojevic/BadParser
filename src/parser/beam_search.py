from parser.configuration import Configuration
from parser.action import ActionStorage
from math import isfinite
import heapq
import numpy as np
import dynet as dy

from data_formats.tree_representation import TreeNode

class BeamDecoder:

    def __init__(self, params, w2i, p2i, n2i, beam_size):
        self.action_storage = ActionStorage(n2i, params['E_a'])
        self.params = params
        self.w2i = w2i
        self.p2i = p2i
        self.n2i = n2i
        self.beam_size = beam_size


    def decode(self, words, pos_seq): # <<<<<<>>>>>>
        dy.renew_cg()
        init_conf = \
            Configuration.construct_init_configuration(
                words, pos_seq, self.params, self.action_storage, self.w2i, self.p2i)
        current_beam = [init_conf]

        best_finished_conf = None
        best_finished_conf_log_prob = -float('inf')

        while not self.whole_beam_finished(current_beam):
            options = []
            for c in current_beam:
                if c.is_final_configuration():
                    if best_finished_conf_log_prob < c.log_prob.value():
                        best_finished_conf = c
                        best_finished_conf_log_prob = c.log_prob.value()
                else:
                    log_probs = c.action_log_probabilities().npvalue()
                    for i in range(len(log_probs)):
                        if isfinite(log_probs[i]) and log_probs[i] > best_finished_conf_log_prob:
                            options.append((c, i, c.log_prob.value()+log_probs[i]))
            kbest_options = heapq.nlargest(self.beam_size, options, key=lambda x:x[2])
            new_beam = []
            for c, t, _ in kbest_options:
                new_beam.append(c.transition(t))
            current_beam = new_beam

        for c in current_beam:
            if best_finished_conf_log_prob < c.log_prob.value():
                best_finished_conf = c
                best_finished_conf_log_prob = c.log_prob.value()

        tree = best_finished_conf.stack.top()

        if tree.label != "root":
            tree = TreeNode("root", [tree], {})

        return best_finished_conf, tree


    def whole_beam_finished(self, beam):
        for c in beam:
            if not c.is_final_configuration():
                return False
        return True

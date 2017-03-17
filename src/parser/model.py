import json
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect
from random import shuffle
import dynet as dy
import re

from data_formats.tree_loader import load_from_export_format
from parser.string2int_mapper import String2IntegerMapper
from parser.configuration import Configuration
from parser.reccurrent_builder_wrapper import LSTMBuilderWrapper
from parser.action import Action, ActionStorage
from parser.beam_search import BeamDecoder

def define_model(hyper_params):
    model = dy.Model()

    params = dict()

    params['E_w'] = model.add_lookup_parameters((hyper_params['w_voc_size'], hyper_params['w_emb_size']))
    params['E_p'] = model.add_lookup_parameters((hyper_params['p_voc_size'], hyper_params['p_emb_size']))
    params['E_n'] = model.add_lookup_parameters((hyper_params['n_voc_size'], hyper_params['n_emb_size']))
    params['E_a'] = model.add_lookup_parameters((hyper_params['a_voc_size'], hyper_params['a_emb_size']))

    input_rep_size = hyper_params['w_emb_size']+hyper_params['p_emb_size']

    # FFN for terminal transformation
    params['V'] = model.add_parameters((hyper_params['node_rep_size'], input_rep_size))
    params['v'] = model.add_parameters(hyper_params['node_rep_size'])

    conf_input_rep_size = hyper_params['stack_hidden_size'] + hyper_params['buffer_hidden_size'] + hyper_params['action_hidden_size']
    params['W'] = model.add_parameters((hyper_params['config_rep_size'], conf_input_rep_size))
    params['w'] = model.add_parameters(hyper_params['config_rep_size'])

    params['Stack_LSTM'] = LSTMBuilderWrapper(
        hyper_params['stack_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['stack_hidden_size'],  # output size
        model=model)
    params['Buffer_LSTM'] = LSTMBuilderWrapper(
        hyper_params['buffer_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['buffer_hidden_size'], # output size
        model=model)
    params['Action_LSTM'] = LSTMBuilderWrapper(
        hyper_params['action_lstm_layers'],
        hyper_params['a_emb_size'],          # input  size
        hyper_params['action_hidden_size'],  # output size
        model=model)

    # RecursiveNN for trees
    params['U_adj'] = model.add_parameters((hyper_params['node_rep_size'], 2*hyper_params['node_rep_size']+1))
    params['u_adj'] = model.add_parameters(hyper_params['node_rep_size'])
    params['U_pro'] = model.add_parameters((hyper_params['node_rep_size'], hyper_params['node_rep_size']+hyper_params['n_emb_size']))
    params['u_pro'] = model.add_parameters(hyper_params['node_rep_size'])

    # FFN for next action
    params['G'] = model.add_parameters((hyper_params['a_voc_size'], hyper_params['config_rep_size']))
    params['g'] = model.add_parameters(hyper_params['a_voc_size'])

    return model, params

def load_hyper_parameters_from_file(file):
    with open(file) as fh:
        hyper_params = json.load(fh)
    return hyper_params

def load_hyper_parameters_from_model_dir(model_dir):
    return load_hyper_parameters_from_file(join(model_dir, "hyper_parameters.json"))

def load_model(model_dir):
    model = dy.Model()

    components = model.load(join(model_dir, "parameters"))

    hyper_params = load_hyper_parameters_from_model_dir(model_dir)

    params = dict()
    params['E_w'] = components[0]
    params['E_p'] = components[1]
    params['E_n'] = components[2]
    params['E_a'] = components[3]
    params['V'] = components[4]
    params['v'] = components[5]
    params['W'] = components[6]
    params['w'] = components[7]
    params['U_adj'] = components[8]
    params['u_adj'] = components[9]
    params['U_pro'] = components[10]
    params['u_pro'] = components[11]
    params['G'] = components[12]
    params['g'] = components[13]

    builder = components[14]
    params['Stack_LSTM'] = LSTMBuilderWrapper(
        hyper_params['stack_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['stack_hidden_size'],  # output size
        builder=builder)

    builder = components[15]
    params['Buffer_LSTM'] = LSTMBuilderWrapper(
        hyper_params['buffer_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['buffer_hidden_size'], # output size
        builder=builder)

    builder = components[16]
    params['Action_LSTM'] = LSTMBuilderWrapper(
        hyper_params['action_lstm_layers'],
        hyper_params['a_emb_size'],          # input  size
        hyper_params['action_hidden_size'],  # output size
        builder=builder)

    w2i = String2IntegerMapper.load(join(model_dir, "w2i"))
    p2i = String2IntegerMapper.load(join(model_dir, "p2i"))
    n2i = String2IntegerMapper.load(join(model_dir, "n2i"))

    return model, params, hyper_params, w2i, p2i, n2i

def save_model(model, params, hyper_params, model_dir, w2i, p2i, n2i):
    list_of_params = [
        params['E_w'],
        params['E_p'],
        params['E_n'],
        params['E_a'],
        params['V'],
        params['v'],
        params['W'],
        params['w'],
        params['U_adj'],
        params['u_adj'],
        params['U_pro'],
        params['u_pro'],
        params['G'],
        params['g'],
        params['Stack_LSTM'].builder,
        params['Buffer_LSTM'].builder,
        params['Action_LSTM'].builder
    ]

    model.save(join(model_dir, "parameters"), list_of_params)

    w2i.save(join(model_dir, "w2i"))
    p2i.save(join(model_dir, "p2i"))
    n2i.save(join(model_dir, "n2i"))

    with open(join(model_dir, "hyper_parameters.json"), "w") as fh:
        json.dump(hyper_params, fh)

def load_sentences(file):
    all_sentences = []
    with open(file) as fh:
        for line in fh:
            words = re.split("\s+", line)
            all_sentences.append(words)
    return all_sentences


def load_pos_tags(file):
    all_pos_tags = []
    with open(file) as fh:
        for line in fh:
            pos_tags = []
            fields = re.split("\s+", line.rstrip())
            for field in fields:
                sub_fields = field.split("_") # / is a Stanford tagger separator
                assert(len(sub_fields)>1)
                pos_tags.append(sub_fields[-1])
            all_pos_tags.append(pos_tags)
    return all_pos_tags



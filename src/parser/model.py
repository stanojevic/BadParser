import json
from os import listdir, mkdir, pardir
from os.path import join, exists
from sys import stderr
import os, sys, inspect
from random import shuffle
import dynet as dy
import re
import numpy as np

from data_formats.tree_loader import load_from_export_format
from parser.string2int_mapper import String2IntegerMapper, ContainerStr2IntMaps
from parser.configuration import Configuration
from parser.reccurrent_builder_wrapper import LSTMBuilderWrapper
from parser.action import Action, ActionStorage
from parser.beam_search import BeamDecoder

def _use_char_embeddings(hyper_params):
    if 'c_emb_size_for_char' in hyper_params and hyper_params['c_emb_size_for_char'] > 0 and hyper_params['c_emb_size_for_word'] > 0:
        return True
    else:
        return False

def _use_bilstm(hyper_params):
    if 'bilstm_layers' in hyper_params and hyper_params['bilstm_layers'] > 0:
        return True
    else:
        return False

def _use_pretrained_embeddings(hyper_params):
    if 'use_pretrained_emb' in hyper_params and hyper_params['use_pretrained_emb'] != 0:
        return True
    else:
        return False

def define_model(hyper_params, all_s2i, external_embeddings_file=None):
    model = dy.Model()

    params = dict()

    params['E_w'] = model.add_lookup_parameters((hyper_params['w_voc_size'], hyper_params['w_emb_size']))
    params['E_p'] = model.add_lookup_parameters((hyper_params['p_voc_size'], hyper_params['p_emb_size']))
    params['E_n'] = model.add_lookup_parameters((hyper_params['n_voc_size'], hyper_params['n_emb_size']))
    params['E_a'] = model.add_lookup_parameters((hyper_params['a_voc_size'], hyper_params['a_emb_size']))

    input_rep_size = 0

    if _use_pretrained_embeddings(hyper_params):
        wpre2i, pretrained_embeddings = load_embeddings(external_embeddings_file)
        all_s2i.ext_w2i = wpre2i
        wpre_voc_size = pretrained_embeddings.shape[0]
        wpre_dimension = pretrained_embeddings.shape[1]
        params['E_pretrained'] = model.add_lookup_parameters((wpre_voc_size, wpre_dimension))
        params['E_pretrained'].init_from_array(pretrained_embeddings)
        input_rep_size += wpre_dimension

    if _use_char_embeddings(hyper_params):

        params['E_c'] = model.add_lookup_parameters((hyper_params['c_voc_size'], hyper_params['c_emb_size_for_char']))

        params['Char_LSTM_Forward'] = LSTMBuilderWrapper(
            1,
            hyper_params['c_emb_size_for_char'],
            hyper_params['c_emb_size_for_word']/2,
            model=model)

        params['Char_LSTM_Backward'] = LSTMBuilderWrapper(
            1,
            hyper_params['c_emb_size_for_char'],
            hyper_params['c_emb_size_for_word']/2,
            model=model)
        input_rep_size += hyper_params['c_emb_size_for_word']

    input_rep_size += hyper_params['w_emb_size']
    input_rep_size += hyper_params['p_emb_size']

    if _use_bilstm(hyper_params):
        params['BiLSTM'] = dy.BiRNNBuilder(
            hyper_params['bilstm_layers'],
            hyper_params['node_rep_size'],
            hyper_params['node_rep_size'],
            model,
            dy.LSTMBuilder)

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
    if 'stack_dropout' in hyper_params:
        params['Stack_LSTM'].set_dropout(hyper_params['stack_dropout'])
    params['Buffer_LSTM'] = LSTMBuilderWrapper(
        hyper_params['buffer_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['buffer_hidden_size'], # output size
        model=model)
    if 'buffer_dropout' in hyper_params:
        params['Buffer_LSTM'].set_dropout(hyper_params['buffer_dropout'])
    params['Action_LSTM'] = LSTMBuilderWrapper(
        hyper_params['action_lstm_layers'],
        hyper_params['a_emb_size'],          # input  size
        hyper_params['action_hidden_size'],  # output size
        model=model)
    if 'action_dropout' in hyper_params:
        params['Action_LSTM'].set_dropout(hyper_params['action_dropout'])

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
    all_s2i = ContainerStr2IntMaps()

    params = dict()
    comp_index = 0
    params['E_w'] = components[comp_index] ; comp_index+=1
    params['E_p'] = components[comp_index] ; comp_index+=1
    params['E_n'] = components[comp_index] ; comp_index+=1
    params['E_a'] = components[comp_index] ; comp_index+=1
    params['V'] = components[comp_index] ; comp_index+=1
    params['v'] = components[comp_index] ; comp_index+=1
    params['W'] = components[comp_index] ; comp_index+=1
    params['w'] = components[comp_index] ; comp_index+=1
    params['U_adj'] = components[comp_index] ; comp_index+=1
    params['u_adj'] = components[comp_index] ; comp_index+=1
    params['U_pro'] = components[comp_index] ; comp_index+=1
    params['u_pro'] = components[comp_index] ; comp_index+=1
    params['G'] = components[comp_index] ; comp_index+=1
    params['g'] = components[comp_index] ; comp_index+=1

    builder = components[comp_index] ; comp_index+=1
    params['Stack_LSTM'] = LSTMBuilderWrapper(
        hyper_params['stack_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['stack_hidden_size'],  # output size
        builder=builder)

    builder = components[comp_index] ; comp_index+=1
    params['Buffer_LSTM'] = LSTMBuilderWrapper(
        hyper_params['buffer_lstm_layers'],
        hyper_params['node_rep_size'],      # input  size
        hyper_params['buffer_hidden_size'], # output size
        builder=builder)

    builder = components[comp_index] ; comp_index+=1
    params['Action_LSTM'] = LSTMBuilderWrapper(
        hyper_params['action_lstm_layers'],
        hyper_params['a_emb_size'],          # input  size
        hyper_params['action_hidden_size'],  # output size
        builder=builder)

    if _use_bilstm(hyper_params):
        params['BiLSTM'] = components[comp_index] ; comp_index+=1

    if _use_char_embeddings(hyper_params):
        params['E_c'] = components[comp_index] ; comp_index+=1
        builder = components[comp_index] ; comp_index+=1
        params['Char_LSTM_Forward'] = LSTMBuilderWrapper(
            1,
            hyper_params['c_emb_size_for_char'],
            hyper_params['c_emb_size_for_word']/2,
            builder=builder)

        builder = components[comp_index] ; comp_index+=1
        params['Char_LSTM_Backward'] = LSTMBuilderWrapper(
            1,
            hyper_params['c_emb_size_for_char'],
            hyper_params['c_emb_size_for_word']/2,
            builder=builder)
        all_s2i.c2i = String2IntegerMapper.load(join(model_dir, "c2i"))

    if _use_pretrained_embeddings(hyper_params):
        params['E_pretrained'] = components[comp_index] ; comp_index+=1
        all_s2i.ext_w2i = String2IntegerMapper.load(join(model_dir, "ext_w2i"))

    all_s2i.w2i = String2IntegerMapper.load(join(model_dir, "w2i"))
    all_s2i.p2i = String2IntegerMapper.load(join(model_dir, "p2i"))
    all_s2i.n2i = String2IntegerMapper.load(join(model_dir, "n2i"))

    return model, params, hyper_params, all_s2i

def save_model(model, params, hyper_params, model_dir, all_s2i):
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

    if _use_bilstm(hyper_params):
        list_of_params.append(params['BiLSTM'])

    if _use_char_embeddings(hyper_params):
        list_of_params.append(params['E_c'])
        list_of_params.append(params['Char_LSTM_Forward'].builder)
        list_of_params.append(params['Char_LSTM_Backward'].builder)
        all_s2i.c2i.save(join(model_dir, "c2i"))

    if _use_pretrained_embeddings(hyper_params):
        list_of_params.append(params['E_pretrained'])
        all_s2i.ext_w2i.save(join(model_dir, "ext_w2i"))


    model.save(join(model_dir, "parameters"), list_of_params)

    if all_s2i.w2i is not None:
        all_s2i.w2i.save(join(model_dir, "w2i"))
    if all_s2i.p2i is not None:
        all_s2i.p2i.save(join(model_dir, "p2i"))
    if all_s2i.n2i is not None:
        all_s2i.n2i.save(join(model_dir, "n2i"))

    with open(join(model_dir, "hyper_parameters.json"), "w") as fh:
        json.dump(hyper_params, fh)

def load_sentences(file):
    all_sentences = []
    with open(file) as fh:
        for line in fh:
            words = re.split("\s+", line)
            all_sentences.append(words)
    return all_sentences


def load_embeddings(file):
    w2i = String2IntegerMapper()
    #w2i.add_string(String2IntegerMapper.UNK)
    embeddings_list = []
    with open(file) as fh:
        for line in fh:
            fields = line.rstrip().split("\t")
            word = fields[0]
            w2i.add_string(word)
            embeddings_list.append(list(map(float, fields[1:])))
    return w2i, np.array(embeddings_list)

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



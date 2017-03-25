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
from parser.reccurrent_builder_wrapper import LSTMBuilderWrapper, NGramBuilderNetwork
from parser.action import Action, ActionStorage
from parser.beam_search import BeamDecoder
from parser.composition_functions import RecursiveNN, LeSTM, TreeLSTM, HeadOnly

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

def _use_configuration_lstm(hyper_params):
    if 'use_configuration_lstm' in hyper_params and hyper_params['use_configuration_lstm'] != 0:
        return True
    else:
        return False

def _use_non_pretrained_w_embeddings(hyper_params):
    if 'w_emb_size' in hyper_params and hyper_params['w_emb_size'] > 0:
        return True
    else:
        return False

def define_model(hyper_params, all_s2i, external_embeddings_file=None):
    model = dy.Model()

    params = dict()

    if _use_non_pretrained_w_embeddings(hyper_params):
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
        if "update_pretrained_emb" in hyper_params and hyper_params["update_pretrained_emb"]:
            params['E_pretrained_update'] = True
        else:
            params['E_pretrained_update'] = False
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

    if 'w_emb_size' in hyper_params:
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

    if _use_configuration_lstm(hyper_params):
        params['ConfigurationLSTM'] = dy.LSTMBuilder(1, hyper_params['config_rep_size'], hyper_params['config_rep_size'], model)

    ################ stack lstm ################################
    if "stack_ngram_count" in hyper_params and hyper_params["stack_ngram_count"]>=0:
        actual_stack_rep_size = hyper_params["stack_ngram_count"]*hyper_params['node_rep_size']
        if actual_stack_rep_size != hyper_params['stack_hidden_size']:
            raise Exception("stack hidden size doesn't fit with stack n-gram count")
        params['Stack_LSTM'] = \
            NGramBuilderNetwork(hyper_params["stack_ngram_count"], hyper_params['node_rep_size'])
    else:
        params['Stack_LSTM'] = LSTMBuilderWrapper(
            hyper_params['stack_lstm_layers'],
            hyper_params['node_rep_size'],      # input  size
            hyper_params['stack_hidden_size'],  # output size
            model=model)
        if 'stack_dropout' in hyper_params:
            params['Stack_LSTM'].set_dropout(hyper_params['stack_dropout'])

    ################ buffer lstm ################################
    if "buffer_ngram_count" in hyper_params and hyper_params["buffer_ngram_count"]>=0:
        actual_buffer_rep_size = hyper_params["buffer_ngram_count"]*hyper_params['node_rep_size']
        if actual_buffer_rep_size != hyper_params['buffer_hidden_size']:
            raise Exception("buffer hidden size doesn't fit with buffer n-gram count")
        params['Buffer_LSTM'] = \
            NGramBuilderNetwork(hyper_params["buffer_ngram_count"], hyper_params['node_rep_size'])
    else:
        params['Buffer_LSTM'] = LSTMBuilderWrapper(
            hyper_params['buffer_lstm_layers'],
            hyper_params['node_rep_size'],      # input  size
            hyper_params['buffer_hidden_size'], # output size
            model=model)
        if 'buffer_dropout' in hyper_params:
            params['Buffer_LSTM'].set_dropout(hyper_params['buffer_dropout'])

    ################ action lstm ################################
    if "action_ngram_count" in hyper_params and hyper_params["action_ngram_count"]>=0:
        actual_action_rep_size = hyper_params["action_ngram_count"]*hyper_params['node_rep_size']
        if actual_action_rep_size != hyper_params['action_hidden_size']:
            raise Exception("action hidden size doesn't fit with action n-gram count")
        params['Action_LSTM'] = \
            NGramBuilderNetwork(hyper_params["action_ngram_count"], hyper_params['node_rep_size'])
    else:
        params['Action_LSTM'] = LSTMBuilderWrapper(
            hyper_params['action_lstm_layers'],
            hyper_params['a_emb_size'],          # input  size
            hyper_params['action_hidden_size'],  # output size
            model=model)
        if 'action_dropout' in hyper_params:
            params['Action_LSTM'].set_dropout(hyper_params['action_dropout'])

    # RecursiveNN for trees
    if hyper_params["composition_function"] == "RecursiveNN":
        params['composition_function'] = RecursiveNN(hyper_params['node_rep_size'], hyper_params['n_emb_size'], model)
    elif hyper_params["composition_function"] == "LeSTM":
        params['composition_function'] = LeSTM(hyper_params['node_rep_size'], hyper_params['n_emb_size'], model)
    elif hyper_params["composition_function"] == "TreeLSTM":
        params['composition_function'] = TreeLSTM(hyper_params['node_rep_size'], hyper_params['n_emb_size'], model)
    elif hyper_params["composition_function"] == "HeadOnly":
        params['composition_function'] = HeadOnly(hyper_params['node_rep_size'], hyper_params['n_emb_size'], model)
    else:
        raise Exception("Unknowon composition functions %s"%hyper_params["composition_function"])
    params['composition_function'].set_dropout(hyper_params["composition_function_dropout"])

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
    if _use_non_pretrained_w_embeddings(hyper_params):
        params['E_w'] = components[comp_index] ; comp_index+=1
    params['E_p'] = components[comp_index] ; comp_index+=1
    params['E_n'] = components[comp_index] ; comp_index+=1
    params['E_a'] = components[comp_index] ; comp_index+=1
    params['V'] = components[comp_index] ; comp_index+=1
    params['v'] = components[comp_index] ; comp_index+=1
    params['W'] = components[comp_index] ; comp_index+=1
    params['w'] = components[comp_index] ; comp_index+=1
    params['G'] = components[comp_index] ; comp_index+=1
    params['g'] = components[comp_index] ; comp_index+=1
    params['composition_function'] = components[comp_index] ; comp_index+=1

    if "stack_ngram_count" in hyper_params:
        params['Stack_LSTM'] = NGramBuilderNetwork(hyper_params["stack_ngram_count"], hyper_params['node_rep_size'])
    else:
        builder = components[comp_index] ; comp_index+=1
        params['Stack_LSTM'] = LSTMBuilderWrapper(
            hyper_params['stack_lstm_layers'],
            hyper_params['node_rep_size'],      # input  size
            hyper_params['stack_hidden_size'],  # output size
            builder=builder)

    if "buffer_ngram_count" in hyper_params:
        params['Buffer_LSTM'] = NGramBuilderNetwork(hyper_params["buffer_ngram_count"], hyper_params['node_rep_size'])
    else:
        builder = components[comp_index] ; comp_index+=1
        params['Buffer_LSTM'] = LSTMBuilderWrapper(
            hyper_params['buffer_lstm_layers'],
            hyper_params['node_rep_size'],      # input  size
            hyper_params['buffer_hidden_size'], # output size
            builder=builder)

    if "action_ngram_count" in hyper_params:
        params['Action_LSTM'] = NGramBuilderNetwork(hyper_params["action_ngram_count"], hyper_params['node_rep_size'])
    else:
        builder = components[comp_index] ; comp_index+=1
        params['Action_LSTM'] = LSTMBuilderWrapper(
            hyper_params['action_lstm_layers'],
            hyper_params['a_emb_size'],          # input  size
            hyper_params['action_hidden_size'],  # output size
            builder=builder)

    if _use_configuration_lstm(hyper_params):
        params['ConfigurationLSTM'] = components[comp_index] ; comp_index+=1

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
        if "update_pretrained_emb" in hyper_params and hyper_params["update_pretrained_emb"]:
            params['E_pretrained_update'] = True
        else:
            params['E_pretrained_update'] = False
        all_s2i.ext_w2i = String2IntegerMapper.load(join(model_dir, "ext_w2i"))

    all_s2i.w2i = String2IntegerMapper.load(join(model_dir, "w2i"))
    all_s2i.p2i = String2IntegerMapper.load(join(model_dir, "p2i"))
    all_s2i.n2i = String2IntegerMapper.load(join(model_dir, "n2i"))

    return model, params, hyper_params, all_s2i

def save_model(model, params, hyper_params, model_dir, all_s2i):
    list_of_params = []
    if _use_non_pretrained_w_embeddings(hyper_params):
        list_of_params.append(params['E_w'])
    list_of_params.append(params['E_p'])
    list_of_params.append(params['E_n'])
    list_of_params.append(params['E_a'])
    list_of_params.append(params['V'])
    list_of_params.append(params['v'])
    list_of_params.append(params['W'])
    list_of_params.append(params['w'])
    list_of_params.append(params['G'])
    list_of_params.append(params['g'])
    list_of_params.append(params['composition_function'])

    if "stack_ngram_count" not in hyper_params:
        list_of_params.append(params['Stack_LSTM'].builder)
    if "buffer_ngram_count" not in hyper_params:
        list_of_params.append(params['Buffer_LSTM'].builder)
    if "action_ngram_count" not in hyper_params:
        list_of_params.append(params['Action_LSTM'].builder)

    if _use_configuration_lstm(hyper_params):
        list_of_params.append(params['ConfigurationLSTM'])


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



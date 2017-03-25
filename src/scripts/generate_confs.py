import itertools



def createAllConfigs():

    conf_dict = { \
        "word_dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "embedding_dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5],

        #"hidden_size_action": [50, 100], \
        #"hidden_size_stack": [50, 100],\
        #"hidden_size_action": [50, 100], \
        #"node_rep_size": range(30, 80+1, 10), \
        #"conf_rep_size": [50, 100],\
        #"recurrent_conf": [0, 1],\
        #"word_ext_size": [30, 100],\
        #"char_emb_dize": [25, 50], \

        #Layers:
        "bilstm_layers":       [2],      #range(3),\
        "stack_lstm_layers":   [2],      #range(3),\
        "buffer_lstm_layers":  [2],      #range(3),\
        "action_lstm_layers":  [2]       #range(3),


        #Fixed:
        #..

    }




    confs = conf_dict.values()
    keys = conf_dict.keys()
    all_confs= list(itertools.product(*confs))


    return all_confs, keys



def writeConfigFiles(dirname, prefix):


    all_confs, keys = createAllConfigs()

    for idx, conf in enumerate(all_confs):
        fh = open("{}/{}{}.json".format(dirname, prefix, idx), "w")

        fh.write("{\n")
        for k,v in zip(keys, conf):
            fh.write("\"{}\":{},\n".format(k, '\"'+v+'\"' if type(v) == str else v))
        fh.write("}")
        fh.close()




if __name__ == "__main__":
    writeConfigFiles("../test", "Config")

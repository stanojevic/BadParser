import itertools



def createAllConfigs():


    word_dropout=[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #embedding_dropout=...
    #...
    recurrent_conf=[0,1]

    #...



    confs = [word_dropout, recurrent_conf]
    keys = ["word_dropout", "recurrent_conf"]
    all_confs= list(itertools.product(*confs))


    return all_confs, keys



def writeConfigFiles(dirname, prefix):


    all_confs, keys = createAllConfigs()

    for idx, conf in enumerate(all_confs):
        fh = open("{}/{}{}.json".format(dirname, prefix, idx), "w")

        #write json format
        fh.write("{\n")
        for k,v in zip(keys, conf):
            fh.write("\"{}\":{},\n".format(k, '\"'+v+'\"' if type(v) == str else v))


        fh.write("}")
        fh.close()




if __name__ == "__main__":
    writeConfigFiles("../test", "Config")
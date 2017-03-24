



def createAllConfigs():

    allParams, keys=config(sim)
    configs= list(itertools.product(*allParams))
    #hpList = [AllMyFields({x:y for x,y in zip(keys, c)}) for c in configs]

    return configs, keys



'''
    Write one configuration file.
'''
def writeConfigFile(fname, config_dict):

    fh = open(fname, "r")

    #write json format


    fh.close()






'''
    Generate bash script to run all the configurations in a certain folder, in the LACO machines.
'''
def writeExecutorScript(fname)

    fh = open(fname, "r")

    #write bash script


    fh.close()

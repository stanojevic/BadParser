BadParser
=======================

BadParser is a transition based neural discontinuous constituency parser. As transitions it uses promote-adjoin transitions for projective part and swap transitions for non-projective constituents. As the probabilistic model it combines recurrent neural networks of different sorts in order to represent that state of the full configuration: there is no part of configuration that is not considered while making the decision about the next action.

Transition-based approach makes this parser very fast and the recurrent neural model makes it very accurate: currently it is the most accurate constituency parser of German.

# Installation

Installing BadParser is relatively simple. From the dependencies it has only DyNet and NumPy. DyNet must be installed with Python bindings as described [here](http://dynet.readthedocs.io/en/latest/python.html).

# Training

BadParser is primarily created for discontinuous parsing. In the domain of discontinuous constituency parsing German is the main test case so BadParser is designed to work with the main file format of the German treebanks Tiger and Negra which are encoded in the Export format. If your data is not in that format you can convert it using [treetools](https://github.com/wmaier/treetools).

BadParser also uses pretrained word embeddings that are important for good performance. These can be extracted with programs like Glove and Word2Vec.

The script that is responsible for training is located in src/scripts/train.py. Its parameters are shown bellow.

Parameter                                                | Description
-------------------------------------------------------- | ----------------------------
-h, --help                                               | shows the help message and exit
--model_dir MODEL_DIR                                    | Model output directory
--train_trees_file TRAIN_TREES_FILE                      | export format training file
--train_pos_file TRAIN_POS_FILE                          | stanford tagger format(sep /) training file
--dev_trees_file DEV_TREES_FILE                          | export format development file
--dev_pos_file DEV_POS_FILE                              | stanford tagger format(sep /) development file
--external_embeddings_file EXTERNAL_EMBEDDINGS_FILE      | csv file with embeddings
--dynet-mem DYNET_MEM                                    | memory for the neural network
--dynet-weight-decay DYNET_WEIGHT_DECAY                  | weight decay (L2) for the neural network
--mini_batch_size MINI_BATCH_SIZE                        | mini-batch size
--encoding ENCODING                                      | Export format encoding default=utf-8, alternative latin1
--epochs EPOCHS                                          | number of epochs
--hyper_params_file HYPER_PARAMS_FILE                    | file with hyperparameters in json format


# Testing

The script that is responsible for testing phase is located in src/scripts/parse.py. Its parameters are shown bellow.

Parameter                                                | Description
-------------------------------------------------------- | ----------------------------
-h, --help                                               | shows the help message and exit
--model_dir MODEL_DIR                                    | Model output directory
--test_sentences_file TEST_SENTENCES_FILE                | sentences to parse
--test_pos_file TEST_POS_FILE                            | stanford tagger format(sep /) test file
--output_file OUTPUT_FILE                                | file to output the trees in
--dynet-mem DYNET_MEM                                    | memory for the neural network
--beam_size BEAM_SIZE                                    | beam size



Authors
-------
[Miloš Stanojević](https://staff.fnwi.uva.nl/m.stanojevic) and [Raquel G. Alhama](https://rgalhama.github.io/)

Institute for Logic, Language and Computation

University of Amsterdam


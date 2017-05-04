from keras.models import Model,Sequential
from keras.layers import Input, Dense, embeddings, merge, Lambda, Flatten,Reshape,convolutional,pooling,recurrent, Permute
from keras.engine.topology import Layer,Merge
import models
import tensorflow as tf
import sys
from keras.models import model_from_json, Model
import helpers
import random
import numpy as np
from scipy.spatial.distance import cosine
from keras import optimizers
from keras.regularizers import l2,l1,l1l2
from keras import backend as K
from keras.constraints import nonneg
import random
import argparse

#constants
MAX_WORD_NGRAMS = 74
NUM_ALL_NGRAMS = 51810
RANDOM_SAMPLES = 100

# process command line arguments
epoch = 0

parser = argparse.ArgumentParser(description='Run substitution algorithm.')
parser.add_argument('type', nargs=1, help='type of model')
parser.add_argument('--e', nargs='?', help='number of epoch to process, default 0')
args = parser.parse_args()

if args.e is not None:
    epoch = int(args.e)

nn_type = args.type[0]

# predict ngram vectors and word vector for given word
def get_input_output_for_word(word,my_model,ngram_keys,ngram_change=None,rand_int=None):
    word_vec = my_model.predict_vector_for_word(word,ngram_keys,ngram_change,rand_int)[0][0]
    if my_model.get_num_inputs()>1:
        ngram_vecs = my_model.predict_ngram_vectors_for_word(word,ngram_keys,ngram_change,rand_int)
        ngram_vecs = np.array(ngram_vecs)
        ngram_vecs = np.concatenate(ngram_vecs,axis=1)
        ngram_vecs = ngram_vecs[0]
    else:
        ngram_vecs = my_model.predict_ngram_vectors_for_word(word,ngram_keys,ngram_change,rand_int)[0]
    ngrams = my_model.create_ngrams_for_word(word,ngram_keys)
    return ngram_vecs,word_vec,ngrams

# substitution algorithm for one word
def count_ngram_weights_for_word(base_word,my_model,ngram_keys):
    keys = word_keys.keys()
    biv,bov,ngrams = get_input_output_for_word(base_word, my_model, ngram_keys)
    ngram_weights = []
    order_weights = []
    for i in range(MAX_WORD_NGRAMS):
        cos_sum = 0
        for j in range(RANDOM_SAMPLES):
            iv,ov,_ = get_input_output_for_word(base_word, my_model, ngram_keys,i,random.randint(2,NUM_ALL_NGRAMS))
            cos_sum += cosine(bov,ov)
        if i < len(ngrams):
            ngram_weights.append((ngrams[i],cos_sum))
        order_weights.append((i,cos_sum))
    return ngram_weights,order_weights

# load data from files
model = helpers.load_model_from_file(epoch=epoch)
ngram_keys = helpers.read_all_ngrams_from_file()
word_keys = helpers.read_all_words_from_file()
my_model = helpers.get_model_by_name(nn_type,model=model)
random_words = helpers.get_10000mixed_words()

# run whole algorithm for all random words, save after each 100
all_ow = []
all_ngw = []
i = 0
for rw in random_words:
    ow,ngw = count_ngram_weights_for_word(rw, my_model, ngram_keys)
    all_ngw.extend(ngw)
    all_ow.extend(ow)
    if i%100==0:
        with open( "subst_all_ngram_weights.txt", "a" ) as out:
            for (k,v) in all_ngw:
                out.write(str(k)+"\t"+str(v)+"\n")

        with open( "subst_all_order_weights.txt", "a" ) as out:
            for (k,v) in all_ow:
                out.write(str(k)+"\t"+str(v)+"\n")
        all_ow = []
        all_ngw = []
    i += 1



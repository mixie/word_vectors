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


epoch = sys.argv[1]
nn_type = sys.argv[2]


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


def count_ngram_weights_for_word(base_word,my_model,ngram_keys):
    keys = word_keys.keys()
    biv,bov,ngrams = get_input_output_for_word(base_word, my_model, ngram_keys)
    ngram_weights = []
    order_weights = []
    for i in range(74):
        cos_sum = 0
        for j in range(100):
            iv,ov,_ = get_input_output_for_word(base_word, my_model, ngram_keys,i,random.randint(2,51810))
            cos_sum += cosine(bov,ov)
        if i < len(ngrams):
            ngram_weights.append((ngrams[i],cos_sum))
        order_weights.append((i,cos_sum))
    return ngram_weights,order_weights


#constants
vec_size = 100
max_ngram_num = 74

model = helpers.load_model_from_file(epoch=epoch)
ngram_keys = helpers.read_all_ngrams_from_file()
word_keys = helpers.read_all_words_from_file()

my_model = helpers.get_model_by_name(nn_type,model=model)

random_words = helpers.get_10000mixed_words()


all_ow = []
all_ngw = []
i = 0
for rw in random_words:
    print "i",i
    ow,ngw = count_ngram_weights_for_word(rw, my_model, ngram_keys)
    all_ngw.extend(ngw)
    all_ow.extend(ow)
    if i%100==0:
        with open( "all_ngw2.txt", "a" ) as out:
            for (k,v) in all_ngw:
                out.write(str(k)+"\t"+str(v)+"\n")

        with open( "all_ow2.txt", "a" ) as out:
            for (k,v) in all_ow:
                out.write(str(k)+"\t"+str(v)+"\n")
        all_ow = []
        all_ngw = []
    i += 1



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
from collections import defaultdict
import pickle
import argparse

# constants
MAX_WORD_NGRAMS = 74
NUM_ALL_NGRAMS = 51810
VECTOR_SIZE = 100
RANDOM_SAMPLES = 500

# process command line args
epoch = 0
parser = argparse.ArgumentParser(description='Run LIME algorithm.')
parser.add_argument('type', nargs=1, help='type of model')
parser.add_argument('--e', nargs='?', help='number of epoch to process, default 0')
args = parser.parse_args()

if args.e is not None:
    epoch = int(args.e)
nn_type = args.type[0]

# Special layer for weighting ngrams
class WeightLayer(Layer):

    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True,
                                      regularizer=l1(0.1),constraint=nonneg())
        super(WeightLayer, self).build(input_shape)  

    def call(self, x,mask=None):
        res = K.permute_dimensions(x,(0,2,1))
        res =  K.dot(res,self.kernel)
        return res

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[2])

# our defined distance between two ngram vectors
def vector_distance(v1,v2):
    dists = 0
    for i in range(len(v1)):
        cos = cosine(v1[i],v2[i])
        if np.isnan(cos):
            if np.count_nonzero(v1[i]) == 0 and  np.count_nonzero(v2[i]) == 0:
                dists += 0
            else:
                dists += 1
        else:
            dists += cos
    return dists/len(v1)

# predict ngram vectors and word vector for given word
def get_input_output_for_word(word,my_model,ngram_keys):
    word_vec = my_model.predict_vector_for_word(word,ngram_keys)[0][0]
    if my_model.get_num_inputs()>1:
        ngram_vecs = my_model.predict_ngram_vectors_for_word(word,ngram_keys)
        ngram_vecs = np.array(ngram_vecs)
        ngram_vecs = np.concatenate(ngram_vecs,axis=1)
        ngram_vecs = ngram_vecs[0]
    else:
        ngram_vecs = my_model.predict_ngram_vectors_for_word(word,ngram_keys)[0]
    ngrams = my_model.create_ngrams_for_word(word,ngram_keys)
    return ngram_vecs,word_vec,ngrams

# generate samples for word, input vectors (ngram vectors), output vectors (word vectors) and weights
def get_samples_for_word(word, num_samples, word_keys,ngram_keys, my_model):
    base_word = word
    keys = word_keys.keys()
    other_words = helpers.generate_random_english_words(num_samples)
    biv,bov,word_ngrams = get_input_output_for_word(base_word, my_model, ngram_keys)
    inputs = [biv]
    outputs = [bov]
    weights = [1]
    count = 0
    for w in other_words:
        iv,ov,_ = get_input_output_for_word(w, my_model, ngram_keys)
        wg = np.exp(-vector_distance(iv,biv)/10)
        if not np.isnan(wg):
            weights.append(wg)
            inputs.append(iv)
            outputs.append(ov)
            count += 1
    return inputs,outputs,weights,word_ngrams

# create model for LIME alg. and fit it with data
def create_and_fit_model(inputs,outputs,weights,batch_size=10,nb_epoch=8):
    new_model = Sequential()
    new_model.add(WeightLayer(input_shape=(MAX_WORD_NGRAMS,VECTOR_SIZE)))
    new_model.add(Flatten())
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    new_model.compile(loss='cosine_proximity', optimizer=sgd)

    new_model.fit(np.array(inputs),np.array(outputs),batch_size=batch_size,nb_epoch=nb_epoch, sample_weight=np.array(weights))
    return new_model

# extract ngram weights from model
def make_weights_for_ngrams(pred_model,word_ngrams):
    weights = np.array(pred_model.layers[0].get_weights())
    ngram_weights = []
    order_weights = []
    for i in range(len(word_ngrams)):
        ngram_weights.append((word_ngrams[i],weights[0][i][0]))
    for i in range(weights[0].shape[0]):
        order_weights.append((i,weights[0][i][0]))
    return ngram_weights,order_weights

# run whole lime algorithm 
def lime_alg(word, my_model, word_keys, ngram_keys,samples):
    inputs,outputs,weights,word_ngrams = get_samples_for_word(word,samples,word_keys,ngram_keys, my_model)
    pred_model = create_and_fit_model(inputs,outputs,weights)
    ngram_weights,order_weights = make_weights_for_ngrams(pred_model,word_ngrams)
    return word,order_weights, ngram_weights

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
    print "i",i
    w,ow,ngw = lime_alg(rw,my_model,word_keys,ngram_keys,samples=RANDOM_SAMPLES)
    all_ngw.extend(ngw)
    all_ow.extend(ow)
    if i%100==0:
        with open( "lime_all_ngram_weights.txt", "a" ) as out:
            for (k,v) in all_ngw:
                out.write(str(k)+"\t"+str(v)+"\n")

        with open( "lime_all_order_weights.txt", "a" ) as out:
            for (k,v) in all_ow:
                out.write(str(k)+"\t"+str(v)+"\n")
        all_ow = []
        all_ngw = []
    i += 1


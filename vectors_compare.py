import helpers
import pickle
from scipy.spatial.distance import cosine
import numpy as np
import sys
import fileinput
import csv
from keras.models import model_from_json, Model
from keras.engine.topology import Layer
from keras import backend as K
import math, random
from keras.models import Model,Sequential
from keras.layers import Input, Dense, embeddings, merge, Lambda, Flatten,Reshape,convolutional,pooling,recurrent,Activation
from scipy.stats import spearmanr
import argparse

VECTOR_SIZE = 100
MAX_NGRAM_NUM = 74

import_ngrams = None
simple_model = None

simple_konv_model = None

parser = argparse.ArgumentParser(description='Compare our results with datasets.')
parser.add_argument('first', nargs=1, help='1. weight file number')
parser.add_argument('last', nargs=1, help='last weight file number')
parser.add_argument('type', nargs=1, help='type of model')
parser.add_argument('--imp', nargs='?', help='important ngrams file, if applicable')
args = parser.parse_args()

nn_type = args.type[0]
start = int(args.first[0])
num_input_files = int(args.last[0])

if args.imp is not None:
    import_ngrams = helpers.read_ngrams_from_file(args.imp)


datasets = {"rw":"../../../testdata/rw.txt","sim":"../../../testdata/combined.tab"}

with open('model.json', 'r') as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json,custom_objects={"ZeroMaskedEntries": helpers.models.ZeroMaskedEntries})

ngram_keys = {}
i = 0
with open('all_unique_ngrams.txt', 'r') as input1:
    for line in input1:
        if line.startswith("# "):
            continue
        ngram_keys[line.strip()]=i
        i += 1

word_keys = {}
i = 0
with open('all_unique_words.txt', 'r') as input1:
    for line in input1:
        if line.startswith("# "):
            continue
        word_keys[line.strip()]=i
        i += 1

def create_simple_conv_model(my_model):
    model = Sequential()
    conv = convolutional.Convolution1D(VECTOR_SIZE, MAX_NGRAM_NUM, border_mode='valid',input_shape=(MAX_NGRAM_NUM,VECTOR_SIZE))
    model.add(conv)
    model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    conv.set_weights(my_model.keras_model.get_layer(my_model.vector_layer_name).get_weights())
    return model

def predict_with_zeroed_ngrams(my_model,word,ngram_keys,keys2,new_model):
    ngram_vector = my_model.predict_ngram_vectors_for_word(word,ngram_keys)
    ngrams = my_model.create_ngrams_for_word(word,ngram_keys)
    zero_indexes = []
    i = 0
    for ng in ngrams:
        if ng not in keys2:
            zero_indexes.append(i)
        i += 1
    for i in range(MAX_NGRAM_NUM):
        if i in zero_indexes:
            ngram_vector[0,i,:] = 0
    return new_model.predict(ngram_vector)

def compare_to_dataset(dataset_file,ngram_keys,simple_model=None,import_ngrams=None):
    ngram_results = []
    file_data = []
    with open(dataset_file, 'rb') as csvfile:
        csvr = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in csvr:
            if row[0]=="Word 1":
                continue
            if simple_model is not None:
                v1 = cosine(predict_with_zeroed_ngrams(my_model,row[0].strip(),ngram_keys,keys2=import_ngrams,new_model=simple_model), 
                    predict_with_zeroed_ngrams(my_model,row[0].strip(),ngram_keys,keys2=import_ngrams,new_model=simple_model))
            else:
                if isinstance(my_model, helpers.models.WordVectorModel) and (row[0].strip() not in ngram_keys 
                    or row[1].strip() not in ngram_keys):
                    v1 = 0
                else:
                    v1 = cosine(my_model.predict_vector_for_word(row[0].strip(),ngram_keys,keys2=import_ngrams), 
                    my_model.predict_vector_for_word(row[1].strip(),ngram_keys,keys2=import_ngrams))
                    file_data.append(float(row[2]))
                    if math.isnan(v1):
                        v1 = 0
                    ngram_results.append(1-v1)
        r,p = spearmanr(np.array(file_data),np.array(ngram_results))
        return r


my_model = helpers.get_model_by_name(nn_type,model=model)
if import_ngrams is not None:
    simple_model = create_simple_conv_model(my_model)


for i in range(start, num_input_files):
    model.load_weights('weights0_fo'+str(i)+'_fn'+str(i)+'_copy.h5')
    weights = None
    for layer in model.layers:
        if layer.get_config()["name"]=="input_layer":
            weights = layer.get_weights()[0]
            break

    for ds in datasets:
        if isinstance(my_model, helpers.models.WordVectorModel):
            print i,ds,compare_to_dataset(datasets[ds],word_keys)
        else:
            print i,ds,compare_to_dataset(datasets[ds],ngram_keys,simple_model,import_ngrams)



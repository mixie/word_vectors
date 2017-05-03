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
import helpers

#constants
vec_size = 100
num_input_files = int(sys.argv[1])
nn_type = sys.argv[2] #ngram_sum
max_ngram_num = 15


#python vectors_compare.py ../preprocessed/preprocessed/ 5 ngram_sum 0.1


datasets = {"rw":"/mnt/data/diplomka/testdata/questions-words.txt"}

#read model structure
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json,custom_objects={"ZeroMaskedEntries": helpers.ZeroMaskedEntries})

#read ngram keys
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

my_model = helpers.get_model_by_name(nn_type,model=model)

#funkcia, ktora porovna natrenovane hodnoty s danym datasetom
def compare_to_dataset(dataset_file,ngram_keys):
    all_ngrams = []
    input_data = []
    with open(dataset_file, 'rb') as csvfile:
        csvr = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count_good = 0
        for row in csvr:
            if row[0].startswith(":"):
                continue
            v1 = my_model.predict_vector_for_word(row[0].strip(),ngram_keys)
            v2 = my_model.predict_vector_for_word(row[1].strip(),ngram_keys)
            v3 = my_model.predict_vector_for_word(row[2].strip(),ngram_keys)

            v = v1 - v2 + v3

            v4 = my_model.predict_vector_for_word(row[3].strip(),ngram_keys)
            #v1 = cosine(vector_f(row[0].strip(),res_ngrams,ngram_keys,None), vector_f(row[1].strip(),res_ngrams,ngram_keys,None))
            res = cosine(v, v4)
            if res < 0.3:
                count_good += 1
            #v2 = cosine(get_vector_for_word(row[0].strip(),res_ngrams,row[1].strip()), get_vector_for_word(row[1].strip(),res_ngrams,row[0].strip()))
            print res, row
        print count_good


#spustenie vsetkeho a vystup
for i in range(num_input_files):
    model.load_weights('weights'+str(i)+'.h5')
    weights = None
    for layer in model.layers:
        if layer.get_config()["name"]=="input_layer":
            weights = layer.get_weights()[0]
            break

    for ds in datasets:
        # if nn_type=="ngram_sum":
        #     print i,ds,compare_to_dataset(datasets[ds],weights,ngram_keys,all_vecs_predictions)
        # if nn_type=="ngram_max":
        #     print i,ds,compare_to_dataset(datasets[ds],weights,ngram_keys,all_vecs_predictions)
        # if nn_type=="classic_vectors":
        if isinstance(my_model, helpers.WordVectorModel):
            print i,ds,compare_to_dataset(datasets[ds],word_keys)
        else:
            print i,ds,compare_to_dataset(datasets[ds],ngram_keys)

          




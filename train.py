from keras.models import Model
from keras.layers import Input, Dense, embeddings, merge, Lambda, Flatten,Reshape,convolutional,pooling
from keras.layers.normalization import BatchNormalization
import numpy as np
import random
from scipy.spatial.distance import cosine
from keras import backend as K
import sys
from keras.engine.topology import Layer
from keras.models import load_model
from keras.optimizers import SGD, TFOptimizer
import os
from datetime import datetime
import tensorflow as tf
from keras.regularizers import l2,l1
import helpers
from models import *
from collections import defaultdict
import datetime

#OMP_NUM_THREADS=1 python train.py ../preprocessed/preprocessed/ input_cleared 50 ngram_max 0 0.1 -1 "skuska" 1 0 0 | tee -a STH.log

#constants
VECTOR_SIZE = 100
BATCH_SIZE = 1
NEGATIVE_SAMPLE_NUM = 5
WINDOW_SIZE = 5
input_folder = sys.argv[1]
input_file_name = sys.argv[2]
num_input_files = int(sys.argv[3])
nn_type = sys.argv[4] #ngram_sum,ngram_conv
epoch_start = int(sys.argv[5])
file_start = int(sys.argv[10])
word_start = int(sys.argv[11])
learning_rate = float(sys.argv[6]) #pre ngram sum 0.05
folder_num = int(sys.argv[7])
poznamka = sys.argv[8]
cpu = int(sys.argv[9])
min_word_freq = 10
min_ngram_freq = 5
special_ngram_file = None
if len(sys.argv)>12:
    special_ngram_file = sys.argv[12]

batch_sizes = {1:10}

max_ngram_num = 74  #http://www.ravi.io/language-word-lengths, max word length -18
folder_num_rand = datetime.datetime.now().strftime('%Y%m%d%H%M')

if folder_num == -1:
    folder_num = folder_num_rand

dummy_model = helpers.get_model_by_name(nn_type, None, None, None, None, None, None, None)


word_keys,words = helpers.generate_word_keys(num_input_files,input_folder,input_file_name,min_word_freq)
if special_ngram_file is not None:
    print "TUUUU"
    ngram_keys,ngrams =  helpers.generate_ngram_keys_from_file(dummy_model,special_ngram_file)
else:
    ngram_keys,ngrams = helpers.generate_ngram_keys(dummy_model,word_keys)


voc_size = len(word_keys)
ngram_voc_size = len(ngram_keys)
print ngram_voc_size

my_model = helpers.get_model_by_name(nn_type,VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num)

my_model.create_model()

if not os.path.exists(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)):
    os.makedirs(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num))

    #write all words to output
    with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/all_unique_words.txt', 'wb') as output:
        output.write("# window: "+str(WINDOW_SIZE)+" min word freq: "+str(min_word_freq)+" min ngram freq: "+str(min_ngram_freq)+'\n')
        for w in words:
            output.write(w+"\n")

    #write all ngrams to output
    with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/all_unique_ngrams.txt', 'wb') as output:
        output.write("# window: "+str(WINDOW_SIZE)+" min word freq: "+str(min_word_freq)+" min ngram freq: "+str(min_ngram_freq)+'\n')
        for ng in ngrams:
            output.write(ng+"\n")

    #create file with parameters
    if folder_num_rand == folder_num:
        with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/params.txt', "w") as file:
            file.write("VECTOR_SIZE"+'\t'+str(VECTOR_SIZE)+"\n")
            file.write("BATCH_SIZE"+'\t'+str(batch_sizes)+"\n")
            file.write("NEGATIVE_SAMPLE_NUM"+'\t'+str(NEGATIVE_SAMPLE_NUM)+"\n")
            file.write("input_folder"+'\t'+input_folder+"\n")
            file.write("input_file_name"+'\t'+input_file_name+"\n")
            file.write("num_input_files"+'\t'+str(num_input_files)+"\n")
            file.write("max_ngram_num"+'\t'+str(max_ngram_num)+"\n")
            file.write("learning_rate"+'\t'+str(learning_rate)+"\n")
            file.write("nn_type"+'\t'+nn_type+"\n")
            file.write("epoch_start"+'\t'+str(epoch_start)+"\n")
            file.write("nn_type"+'\t'+nn_type+"\n")
            file.write("poznamka"+'\t'+poznamka+"\n")
            file.write("konstantne ngramy s pridanym unknown a after\n")

    #create model json file
    json_model = my_model.keras_model.to_json()
    with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/model.json', "w") as json_file:
        json_file.write(json_model)


history = 0
for ep in range(epoch_start,5):

    if ep>0:
        my_model.keras_model.load_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep-1)+'.h5')
    elif word_start>0 or file_start>0:
        my_model.keras_model.load_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights0_fo'+str(file_start)+'_fn'+str(file_start)+'_copy.h5')         


    file_order = 0

    if ep!=epoch_start:
        file_start = 0


    for fn in range(file_start,num_input_files):

        if fn>0:
            BATCH_SIZE = 10
        print "BATCH SIZE",BATCH_SIZE

        with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/epoch.log', "a+") as logfile:
                logfile.write("epoch "+str(ep)+" file "+str(fn)+" processed "+str(file_order)+"/"+str(num_input_files)+"\n")
        print "ep",ep,"processing file", fn
        sys.stdout.flush()

        with open(input_folder+input_file_name+str(fn)+".txt", 'r') as input1:

            act_file = helpers.read_input_file(input1,word_keys)

            model_inputs = []
            for j in range(NEGATIVE_SAMPLE_NUM+1+my_model.get_num_inputs()):
                model_inputs.append([])

            if fn!=file_start:
                word_start = 0

            for i in range(word_start,len(act_file)):

                if act_file[i]=="" or act_file[i]=="___":
                    continue

                for k in range(-WINDOW_SIZE+i,WINDOW_SIZE+i):
                    if k < 0 or k == i or k>=len(act_file) or act_file[k]=="___" or act_file[k] not in word_keys:
                        continue

                    if my_model.get_num_inputs()==1:
                        if isinstance(my_model, WordVectorModel):
                            model_inputs[0].append(my_model.create_input_for_word(act_file[i],word_keys))
                        else:
                            model_inputs[0].append(my_model.create_input_for_word(act_file[i],ngram_keys))
                    else:
                        for ni in range(my_model.get_num_inputs()):
                            inputs = my_model.create_input_for_word(act_file[i],ngram_keys)
                            model_inputs[ni].append(inputs[ni])

                    model_inputs[my_model.get_num_inputs()].append(word_keys[act_file[k]])

                    for j in range(NEGATIVE_SAMPLE_NUM):
                        model_inputs[my_model.get_num_inputs()+1+j].append(word_keys[random.choice(words)])

                if i%BATCH_SIZE==0 and i>0:
                    model_np_arrays = [np.array(l) for l in model_inputs]
                    res = [np.ones((len(model_inputs[0]),1,1)) for l in range(1+NEGATIVE_SAMPLE_NUM)]
                    if cpu==1:
                        with tf.device('/cpu:0'):
                            hist = my_model.keras_model.train_on_batch(model_np_arrays,res)
                    else:
                        hist = my_model.keras_model.train_on_batch(model_np_arrays,res)
                    history += sum(hist)
                    sys.stdout.flush()
                    model_inputs = []
                    for j in range(NEGATIVE_SAMPLE_NUM+1+my_model.get_num_inputs()):
                        model_inputs.append([])

                if i%1000==0 and i > 0:
                    print "ep",ep,"f",fn,"fo",file_order,i,"/"+str(len(act_file))+'\t',history/1000,"\t"
                    history = 0
                    sys.stdout.flush()

                if i%25000==0 and i>0:
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+'.h5')
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+'_copy.h5')
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+"_fo"+str(file_order)+'_fn'+str(fn)+'_copy.h5')
                    sys.stdout.flush()
            file_order += 1








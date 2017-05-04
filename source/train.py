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
import argparse

#constants
VECTOR_SIZE = 100
BATCH_SIZE = 1
NEGATIVE_SAMPLE_NUM = 5
WINDOW_SIZE = 5
MIN_WORD_FREQ = 10
MIN_NGRAM_FREQ = 5
batch_sizes = {1:10}
MAX_NGRAM_NUM = 74 
folder_num_rand = datetime.datetime.now().strftime('%Y%m%d%H%M')


#processing command line args
parser = argparse.ArgumentParser(description='Train model with given parameters')
parser.add_argument('input_folder', nargs=1, help='input folder for train data')
parser.add_argument('input_name', nargs=1, help='input name for train data')
parser.add_argument('num_input_files', nargs=1, help='number of input files')
parser.add_argument('type', nargs=1, help='type of neural network to train')
parser.add_argument('epoch_start', nargs=1, help='epoch to start training from')
parser.add_argument('lr', nargs=1, help='learning rate')
parser.add_argument('folder_num', nargs=1, help='generated folder number')
parser.add_argument('note', nargs=1, help='note - description of training')
parser.add_argument('cpu', nargs=1, help='cpu = 1, gpu = 0')
parser.add_argument('file_start', nargs=1, help='file to start training from')
parser.add_argument('word_start', nargs=1, help='word to start training from')
parser.add_argument('special_ngram_file', nargs='?', help='''use only when ngrams should not be counted, 
                    but loaded from file instead''')

args = parser.parse_args()
input_folder = args.input_folder[0]
input_file_name = args.input_name[0]
num_input_files = args.num_input_files[0]
nn_type = args.type[0]
epoch_start = args.epoch_start[0]
learning_rate = args.lr[0]
folder_num = args.folder_num[0]
poznamka = args.note[0]
cpu = args.cpu[0]
file_start = args.file_start[0]
word_start = args.word_start[0]
special_ngram_file = None

if args.special_ngram_file is not None:
    special_ngram_file = args.special_ngram_file


if folder_num == -1:
    folder_num = folder_num_rand

# generate all ngram and word keys for model 
dummy_model = helpers.get_model_by_name(nn_type, None, None, None, None, None, None, None)

word_keys,words = helpers.generate_word_keys(num_input_files,input_folder,input_file_name,MIN_WORD_FREQ)
if special_ngram_file is not None:
    ngram_keys,ngrams =  helpers.generate_ngram_keys_from_file(dummy_model,special_ngram_file)
else:
    ngram_keys,ngrams = helpers.generate_ngram_keys(dummy_model,word_keys)

# create model structure for training
voc_size = len(word_keys)
ngram_voc_size = len(ngram_keys)

my_model = helpers.get_model_by_name(nn_type,VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,MAX_NGRAM_NUM)

my_model.create_model()

# save basic model data to file
if not os.path.exists(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)):
    os.makedirs(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num))

    #write all words to output
    with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/all_unique_words.txt', 'wb') as output:
        output.write("# window: "+str(WINDOW_SIZE)+" min word freq: "+str(MIN_WORD_FREQ)+" min ngram freq: "+str(MIN_NGRAM_FREQ)+'\n')
        for w in words:
            output.write(w+"\n")

    #write all ngrams to output
    with open(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/all_unique_ngrams.txt', 'wb') as output:
        output.write("# window: "+str(WINDOW_SIZE)+" min word freq: "+str(MIN_WORD_FREQ)+" min ngram freq: "+str(MIN_NGRAM_FREQ)+'\n')
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
            file.write("MAX_NGRAM_NUM"+'\t'+str(max_ngram_num)+"\n")
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

# training of the model 
history = 0
for ep in range(epoch_start,5):
    # load weights if starting from later epoch
    if ep>0:
        my_model.keras_model.load_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep-1)+'.h5')
    elif word_start>0 or file_start>0:
        my_model.keras_model.load_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights0_fo'+str(file_start)+'_fn'+str(file_start)+'_copy.h5')         


    file_order = 0

    if ep!=epoch_start:
        file_start = 0

    # process all files in epoch
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

                # process all words in word window
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
                    # create negative samples
                    for j in range(NEGATIVE_SAMPLE_NUM):
                        model_inputs[my_model.get_num_inputs()+1+j].append(word_keys[random.choice(words)])

                # process accumulated inputs
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

                # print current error
                if i%1000==0 and i > 0:
                    print "ep",ep,"f",fn,"fo",file_order,i,"/"+str(len(act_file))+'\t',history/1000,"\t"
                    history = 0
                    sys.stdout.flush()

                # save current weights to file 
                if i%25000==0 and i>0:
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+'.h5')
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+'_copy.h5')
                    my_model.keras_model.save_weights(input_folder+'results_'+nn_type+"_"+str(learning_rate)+"_"+str(folder_num)+'/weights'+str(ep)+"_fo"+str(file_order)+'_fn'+str(fn)+'_copy.h5')
                    sys.stdout.flush()
            file_order += 1








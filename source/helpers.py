import random
from keras.engine.topology import Layer
from keras import backend as K
from keras.models import model_from_json
import models
from collections import defaultdict

# read ngrams from text file, ngrams should be one per line
def read_all_ngrams_from_file(path=""):
    ngram_keys = {}
    i = 0
    with open(path+'all_unique_ngrams.txt', 'r') as input1:
        for line in input1:
            if line.startswith("# "):
                continue
            ngram_keys[line.strip()]=i
            i += 1
    return ngram_keys

# read ngrams from text file, words should be one per line
def read_all_words_from_file(path=""):
    word_keys = {}
    i = 0
    with open(path+'all_unique_words.txt', 'r') as input1:
        for line in input1:
            if line.startswith("#"):
                continue
            word_keys[line.strip()]=i
            i += 1
    return word_keys

# load pretrained model from file, might also load model weights
def load_model_from_file(path="",load_weights=True,epoch=0):
    with open(path+'model.json', 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json,custom_objects={"ZeroMaskedEntries": models.ZeroMaskedEntries})
        if load_weights:
            model.load_weights('weights'+str(epoch)+'_copy.h5')
        return model
    return None

# load model class by its text name
def get_model_by_name(name, VECTOR_SIZE=None,NEGATIVE_SAMPLE_NUM=5,
    voc_size=None,learning_rate=None,ngram_voc_size=None,max_ngram_num=74, model=None):
    my_model = None

    if name=="classic_vectors":
        my_model = models.WordVectorModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate, model=model)

    if name=="ngram_sum":
        my_model = models.NgramSumModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_max":
        my_model = models.NgramMaxModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv":
        my_model = models.NgramConvPaddedModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv2":
        my_model = models.NgramConvMultModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_old":
        my_model = models.NgramConvShorterFirstModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_old_sorted":
        my_model = models.NgramConvBeginFirstModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_gated":
        my_model = models.NgramConvShorterFirstGatedModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_old_valid":
        my_model = models.NgramConvShorterFirstValidModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_old_valid_whole":
        my_model = models.NgramConvBeginFirstValidModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_conv_old_valid_whole_gated":
        my_model = models.NgramConvBeginFirstValidGatedModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_gru2":
        my_model = models.NgramGRUMultModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_gru3":
        my_model = models.NgramGRUBeginFirstModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if my_model is None:
        raise NameError("Name of model does not exists.")

    return my_model

# read text files (wikipedia) dumps in our preprocessed format
def read_input_file(input1,word_keys):
    act_file = []
    for l in input1:
        if l.startswith('#'):
            continue
        for w in l.strip().split(","):
            if w in word_keys:
                act_file.append(w)
    return act_file

# read all input files and creates dictionary of all words and its ids, words = keys, ids = values
def generate_word_keys(num_input_files,input_folder,input_file_name,min_word_freq):
    word_freq = defaultdict(lambda: 0)

    for fn in range(num_input_files):
        with open(input_folder+input_file_name+str(fn)+".txt", 'r') as input1:
            for l in input1:
                if l.startswith('#'):
                    continue
                for w in l.strip().split(","):
                    word_freq[w] += 1
    word_freq_tups = sorted([(v,k) for k,v in word_freq.iteritems() if v > min_word_freq and len(k)<=18],reverse=True)
    word_keys = {}
    words = []
    i = 0
    for (v,w) in word_freq_tups:
        word_keys[w] = i
        words.append(w)
        i += 1
    return word_keys,words

# return all ngrams for given word
def get_ngrams_for_word(word):
    word = "^"+word+"$"
    ngrams = []
    for k in range(3,7):
        for i in range(0,len(word)-k+1):
            ngrams.append(word[i:i+k])
    return ngrams

# generate all ngrams for given words, with at least given ngram sequency
def generate_ngram_keys(model,word_keys,min_ngram_freq):
    ngrams_freq = defaultdict(lambda:0)
    for w in word_keys:
        ngrams = get_ngrams_for_word(w)
        for ng in ngrams:
            ngrams_freq[ng] += 1

    ngram_freq_tups = sorted([(v,k) for k,v in ngrams_freq.iteritems() if v > min_ngram_freq],reverse=True)
    (ngram_keys,ngrams,i) = model.get_first_ngram_keys()

    for (v,ng) in ngram_freq_tups:
        ngram_keys[ng] = i
        ngrams.append(ng)
        i += 1
    return ngram_keys,ngrams

# generate random english words, based on two different word sets
def generate_random_english_words(num_words = None):
    words1 = []
    with open("../../testdata/350kwords.txt") as input1:
        for l in input1:
            words1.append(l.strip())
    if num_words is not None:
        return random.sample(words1,num_words/3)
    words2 = []
    with open("../../testdata/wiki-100k.txt") as input1:
        for l in input1:
            words2.append(l.strip())
    if num_words is not None:
        return random.sample(words2,(num_words/3)*2)
    words = words1 + words2
    random.shuffle(words)
    return words

# read previously generated random words to list
def get_10000mixed_words():
    words = []
    with open("../../testdata/10000mixed.txt", "r") as input1:
        for l in input1:
            words.append(l.strip())
    return words
    
# read "nonstandard" ngrams from file
def read_ngrams_from_file(f):
    ngrams = []
    with open(f, "r") as input1:
        for l in input1:
            ngrams.append(l.strip())
    return ngrams

# generate ngram keys based on ngrams file
def generate_ngram_keys_from_file(model,f):
    (ngram_keys,ngrams,i) = model.get_first_ngram_keys()
    with open(f) as inp:
        for l in inp:
            ngram_keys[l.strip()] = i 
            ngrams.append(l.strip())
            i += 1
    return ngram_keys,ngrams
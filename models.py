from keras.models import Model
from keras.layers import Input, Dense, embeddings, merge, Lambda, Flatten,Reshape,convolutional,pooling,recurrent,Activation
from keras.engine.topology import Layer,Merge
from keras.optimizers import SGD, TFOptimizer
import tensorflow as tf
from keras.regularizers import l2,l1
from helpers import *
from abc import ABCMeta,abstractmethod
import numpy as np

NGRAM_SIZES = [20,19,18,17]
ngram_num = sum(NGRAM_SIZES)

class ZeroMaskedEntries(Layer):

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]
        self.built = True
        super(ZeroMaskedEntries, self).build(input_shape)

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


class VectorModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None, model=None):
        self.vector_size = vector_size
        self.negative_sample_num = negative_sample_num
        self.voc_size = voc_size
        self.learning_rate = learning_rate
        self.keras_model = model
        self.emb_layer = None
        self.vector_layer_model = None
        self.emb_layer_model = None
        self.vector_layer_name = ""
        self.emb_layer_name = "zeromaskedentries_1"
        self.min_ngram_size = 3
        self.max_ngram_size = 6

    @abstractmethod
    def create_input_for_word(self,word,keys,change_ngram=None):
        pass

    @abstractmethod
    def create_emb_layer(self):
        pass

    def get_num_inputs(self):
        return 1
    
    def create_ngrams_for_word(self,word,keys,keys2 = None):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            for i in range(0,len(word)-k+1):
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams

    def create_false_ngrams_for_word(self,word,keys):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            for i in range(0,len(word)-k+1):
                if word[i:i+k] in keys:
                    ngrams.append(word[i:i+k])
        return ngrams

    def get_first_ngram_keys(self):
        return ({"*****":0},["*****"],1)

    def create_model(self):
        (iw,emb_in,vv_iw) = self.create_emb_layer()
        self.emb_layer = emb_in

        ow = Input(shape=(1,),dtype='int32',name="outputword")
        nws = []
        for i in range(self.negative_sample_num):
            nws.append(Input(shape=(1,),dtype='int32',name="neg_sam"+str(i)))
        inputs = iw
        inputs.append(ow)
        inputs.extend(nws)

        emb_ov = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.voc_size,init="uniform")
        vv_ov = emb_ov(ow)

        vv_nss = []
        for i in range(self.negative_sample_num):
            vv_nss.append(emb_ov(nws[i]))
        cos_pos = merge([vv_iw, vv_ov], mode='dot', dot_axes=(2,2))
        cos_negs = []
        for i in range(self.negative_sample_num):
            cos_neg = merge([vv_iw, vv_nss[i]], mode='dot', dot_axes=(2,2))
            cos_negs.append(cos_neg)

        ress = []
        ress.append(Lambda(lambda x: K.sigmoid(x))(cos_pos))
        for i in range(self.negative_sample_num):
            ress.append(Lambda(lambda x: K.sigmoid(-x))(cos_negs[i]))


        self.keras_model = Model(input=inputs, output=ress)
        self.keras_model.compile(loss='binary_crossentropy', optimizer=TFOptimizer(tf.train.GradientDescentOptimizer(self.learning_rate)))

    def predict_vector_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        if self.vector_layer_model is None:   
            self.vector_layer_model = Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.vector_layer_name).output)
        inp = [np.reshape(np.array(self.create_input_for_word(word,keys,change_ngram,rand_int,keys2)),(1,self.max_ngram_num))]
        inp.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            inp.append(np.zeros((1,1)))
        return self.vector_layer_model.predict(inp)

    def predict_ngram_vectors_for_word(self,word,keys,change_ngram=None,rand_int=None):
        if self.emb_layer_model is None:
            self.emb_layer_model = Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.emb_layer_name).output)
        inp = [np.reshape(np.array(self.create_input_for_word(word,keys,change_ngram,rand_int)),(1,self.max_ngram_num))]
        inp.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            inp.append(np.zeros((1,1)))
        return self.emb_layer_model.predict(inp)

 

class WordVectorModel(VectorModel):
    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num, model=None):
        super(WordVectorModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "input_layer"
        self.ngram_size = None
        self.max_ngram_num = None
        
    def create_emb_layer(self):
        iw = Input(shape=(1,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.voc_size,init="uniform",name="input_layer")
        vv_iw = emb_in(iw)

        return ([iw],emb_in,vv_iw)

    def create_ngrams_for_word(self,word):
        pass

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        return keys[word]

    def predict_vector_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        if self.vector_layer_model is None:   
            self.vector_layer_model = Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.vector_layer_name).output)
        inp = [np.reshape(np.array(self.create_input_for_word(word,keys,change_ngram,rand_int,keys2)),(1,1))]
        inp.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            inp.append(np.zeros((1,1)))
        return self.vector_layer_model.predict(inp)

class NgramSumModel(VectorModel):

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num, model=None):
        super(NgramSumModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "lambda_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",name="input_layer",mask_zero=True)
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        sl = Lambda(lambda x:K.expand_dims(K.sum(x,1),1),output_shape=lambda s: (None,1, self.vector_size))
        sum_vv = sl(zero_masked_emd)
        return ([iw],emb_in,sum_vv)

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)
        
class NgramMaxModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num, model=None):
        super(NgramMaxModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "lambda_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size/2, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size/2))
        zero_masked_emd = zm(vv_iw)
        sl = Lambda(lambda x:K.expand_dims(K.concatenate([K.max(x,1),K.min(x,1)]),1),output_shape=lambda s: (None,1, self.vector_size))
        sum_vv = sl(zero_masked_emd)
        return ([iw],emb_in,sum_vv)

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)

class NgramConvPaddedModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvPaddedModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "averagepooling1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",name="input_layer")
        vv_iw = emb_in(iw)
        conv_l = convolutional.Convolution1D(self.vector_size, 30, border_mode='same')
        conv = conv_l(vv_iw)
        pool = pooling.AveragePooling1D(self.max_ngram_num,border_mode="same")
        pool_res = pool(conv)
        return ([iw],emb_in,pool_res)

    def get_first_ngram_keys(self):
        return ({"*UNK*":0,"*AFT*":1},["*UNK*","*AFT*"],2)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            counter = 0
            for i in range(0,len(word)-k+1):
                counter += 1
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
                else:
                    ngrams.append("*UNK*")
            for dif in range(NGRAM_SIZES[k-3]-counter):
                ngrams.append("*AFT*") 
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)

        
class NgramConvShorterFirstModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvShorterFirstModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "averagepooling1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l = convolutional.Convolution1D(self.vector_size, 30, border_mode='same')
        conv = conv_l(zero_masked_emd)
        pool = pooling.AveragePooling1D(self.max_ngram_num,border_mode="same")
        pool_res = pool(conv)
        return ([iw],emb_in,pool_res)

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)

class NgramConvShorterFirstValidModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvShorterFirstValidModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "averagepooling1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l = convolutional.Convolution1D(self.vector_size, 30, border_mode='valid')
        conv = conv_l(zero_masked_emd)
        pool = pooling.AveragePooling1D(45,border_mode='valid')
        pool_res = pool(conv)
        return ([iw],emb_in,pool_res)

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)


class NgramConvBeginFirstValidModel(VectorModel):

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvBeginFirstValidModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "convolution1d_1"
        self.emb_layer_name = "zeromaskedentries_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l = convolutional.Convolution1D(self.vector_size, self.max_ngram_num, border_mode='valid')
        conv = conv_l(zero_masked_emd)
        return ([iw],emb_in,conv)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for i in range(0,len(word)):
            for k in range(self.min_ngram_size,self.max_ngram_size+1):
                if i > len(word)-k:
                    continue
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)


class NgramConvShorterFirstGatedModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvShorterFirstGatedModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "averagepooling1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l = convolutional.Convolution1D(self.vector_size, 30, border_mode='same')
        conv = conv_l(zero_masked_emd)
        sigm_conv = Activation("sigmoid")(conv)
        mult_l = Merge(mode='mul')
        mult = mult_l([conv,sigm_conv])
        pool = pooling.AveragePooling1D(self.max_ngram_num,border_mode="same")
        pool_res = pool(mult)
        return ([iw],emb_in,pool_res)

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)


class NgramConvBeginFirstValidGatedModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvBeginFirstValidGatedModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "convolution1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l1 = convolutional.Convolution1D(self.vector_size, self.max_ngram_num, border_mode='valid')
        conv_l2 = convolutional.Convolution1D(self.vector_size, self.max_ngram_num, border_mode='valid')
        conv1 = conv_l1(zero_masked_emd)
        conv2 = conv_l2(zero_masked_emd)
        sigm_conv = Activation("sigmoid")(conv2)
        mult_l = Merge(mode='mul')
        mult = mult_l([conv1,sigm_conv])
        return ([iw],emb_in,mult)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for i in range(0,len(word)):
            for k in range(self.min_ngram_size,self.max_ngram_size+1):
                if i > len(word)-k:
                    continue
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)

class NgramConvBeginFirstModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvBeginFirstModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "averagepooling1d_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        vv_iw = emb_in(iw)
        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))
        zero_masked_emd = zm(vv_iw)
        conv_l = convolutional.Convolution1D(self.vector_size, 30, border_mode='same')
        conv = conv_l(zero_masked_emd)
        pool = pooling.AveragePooling1D(self.max_ngram_num,border_mode="same")
        pool_res = pool(conv)
        return ([iw],emb_in,pool_res)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for i in range(0,len(word)):
            for k in range(self.min_ngram_size,self.max_ngram_size+1):
                if i > len(word)-k:
                    continue
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)



class NgramConvMultModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramConvMultModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "merge_1"
        self.ngram_size = ngram_size
        self.max_ngram_one_class = 20
        self.max_ngram_num = max_ngram_num


    def get_num_inputs(self):
        return 4

    def create_emb_layer(self):
        iw3 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword3")
        iw4 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword4")
        iw5 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword5")
        iw6 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword6")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        
        vv_iw3 = emb_in(iw3)
        vv_iw4 = emb_in(iw4)
        vv_iw5 = emb_in(iw5)
        vv_iw6 = emb_in(iw6)

        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))

        zero_masked_emd3 = zm(vv_iw3)
        zero_masked_emd4 = zm(vv_iw4)
        zero_masked_emd5 = zm(vv_iw5)
        zero_masked_emd6 = zm(vv_iw6)

        conv_l3 = convolutional.Convolution1D(self.vector_size, 10, border_mode='same')
        conv3 = conv_l3(zero_masked_emd3)
        conv_l4 = convolutional.Convolution1D(self.vector_size, 10, border_mode='same')
        conv4 = conv_l4(zero_masked_emd4)
        conv_l5 = convolutional.Convolution1D(self.vector_size, 10, border_mode='same')
        conv5 = conv_l5(zero_masked_emd5)
        conv_l6 = convolutional.Convolution1D(self.vector_size, 10, border_mode='same')
        conv6 = conv_l6(zero_masked_emd6)
        pool3 = pooling.AveragePooling1D(self.max_ngram_one_class,border_mode="same")
        pool_res3 = pool3(conv3)
        pool4 = pooling.AveragePooling1D(self.max_ngram_one_class,border_mode="same")
        pool_res4 = pool4(conv4)
        pool5 = pooling.AveragePooling1D(self.max_ngram_one_class,border_mode="same")
        pool_res5 = pool5(conv5)
        pool6 = pooling.AveragePooling1D(self.max_ngram_one_class,border_mode="same")
        pool_res6 = pool6(conv6)

        merge_conv = Merge(mode='ave', concat_axis=1)
        merged = merge_conv([pool_res3,pool_res4,pool_res5,pool_res6])
        return ([iw3,iw4,iw5,iw6],emb_in,merged)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            counter = 0
            for i in range(0,len(word)-k+1):
                counter += 1
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams
 
    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        if change_ngram is not None:
            print "ERROR"
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        ngram_keys = [[],[],[],[]]
        for ng in ngrams:
            ngram_keys[len(ng)-3].append(keys[ng])
        for nk in ngram_keys:
            for i in range(self.max_ngram_one_class-len(nk)):
                nk.append(0)
        ngram_keys = [np.array(nk) for nk in ngram_keys]
        return ngram_keys

    def predict_vector_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        if self.vector_layer_model is None:   
            self.vector_layer_model = Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.vector_layer_name).output)
        inp = self.create_input_for_word(word,keys,change_ngram,rand_int,keys2)
        res = []
        for i in inp:
            res.append(np.reshape(i,(1, self.max_ngram_one_class)))
        res.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            res.append(np.zeros((1,1)))
        return self.vector_layer_model.predict(res)

    def predict_ngram_vectors_for_word(self,word,keys,change_ngram=None,rand_int=None):
        if self.emb_layer_model is None:
            self.emb_layer_model = []
            for i in range(self.get_num_inputs()):
                self.emb_layer_model.append(Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.emb_layer_name).get_output_at(i)))
        inp = self.create_input_for_word(word,keys,change_ngram,rand_int)
        inps = []
        for i in inp:
            inps.append(np.reshape(i,(1, self.max_ngram_one_class)))
        inps.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            inps.append(np.zeros((1,1)))
        res = []
        for emb in self.emb_layer_model:
            res.append(emb.predict(inps))
        return res

class NgramGRUModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramGRUModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "???"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",name="input_layer")
        vv_iw = emb_in(iw)
        lstm_l = recurrent.GRU(self.vector_size,return_sequences=False)
        lstm = lstm_l(vv_iw)
        return ([iw],emb_in,lstm)
    
    def get_first_ngram_keys(self):
        return ({"*UNK*":0,"*AFT*":1},["*UNK*","*AFT*"],2)

    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            counter = 0
            for i in range(0,len(word)-k+1):
                counter += 1
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
                else:
                    ngrams.append("*UNK*")
            for dif in range(NGRAM_SIZES[k-3]-counter):
                ngrams.append("*AFT*") 
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)

class NgramGRUMultModel(VectorModel): 

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramGRUMultModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "merge_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num
        self.max_ngram_one_class = 20

    def create_emb_layer(self):
        iw3 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword3")
        iw4 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword4")
        iw5 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword5")
        iw6 = Input(shape=(self.max_ngram_one_class,), dtype='int32', name="inputword6")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        
        vv_iw3 = emb_in(iw3)
        vv_iw4 = emb_in(iw4)
        vv_iw5 = emb_in(iw5)
        vv_iw6 = emb_in(iw6)

        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))

        zero_masked_emd3 = zm(vv_iw3)
        zero_masked_emd4 = zm(vv_iw4)
        zero_masked_emd5 = zm(vv_iw5)
        zero_masked_emd6 = zm(vv_iw6)        

        lstm_l3 = recurrent.GRU(self.vector_size,return_sequences=False)
        lstm_l4 = recurrent.GRU(self.vector_size,return_sequences=False)
        lstm_l5 = recurrent.GRU(self.vector_size,return_sequences=False)
        lstm_l6 = recurrent.GRU(self.vector_size,return_sequences=False)

        lstm3 = lstm_l3(zero_masked_emd3)
        lstm4 = lstm_l4(zero_masked_emd4)
        lstm5 = lstm_l5(zero_masked_emd5)
        lstm6 = lstm_l6(zero_masked_emd6)

        merge_conv = Merge(mode='ave', concat_axis=1)
        merged = merge_conv([lstm3,lstm4,lstm5,lstm6])

        reshaped = Reshape((1,self.vector_size))(merged)
        return ([iw3,iw4,iw5,iw6],emb_in,reshaped)

    def get_num_inputs(self):
        return 4
    
    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for k in range(self.min_ngram_size,self.max_ngram_size+1):
            counter = 0
            for i in range(0,len(word)-k+1):
                counter += 1
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams
 
    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        ngram_keys = [[],[],[],[]]
        if change_ngram is not None:
            print "ERROR"
        for ng in ngrams:
            ngram_keys[len(ng)-3].append(keys[ng])
        for nk in ngram_keys:
            for i in range(self.max_ngram_one_class-len(nk)):
                nk.append(0)
        ngram_keys = [np.array(nk) for nk in ngram_keys]
        return ngram_keys

    def predict_vector_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        if self.vector_layer_model is None:   
            self.vector_layer_model = Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.vector_layer_name).output)
        inp = self.create_input_for_word(word,keys,change_ngram,rand_int,keys2)
        res = []
        for i in inp:
            res.append(np.reshape(i,(1, self.max_ngram_one_class)))
        res.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            res.append(np.zeros((1,1)))
        return self.vector_layer_model.predict(res)

    def predict_ngram_vectors_for_word(self,word,keys,change_ngram=None,rand_int=None):
        if self.emb_layer_model is None:
            self.emb_layer_model = []
            for i in range(self.get_num_inputs()):
                self.emb_layer_model.append(Model(input=self.keras_model.input,
                                 output=self.keras_model.get_layer(self.emb_layer_name).get_output_at(i)))
        inp = self.create_input_for_word(word,keys,change_ngram,rand_int)
        inps = []
        for i in inp:
            inps.append(np.reshape(i,(1, self.max_ngram_one_class)))
        inps.append(np.zeros((1,1)))
        for i in range(self.negative_sample_num):
            inps.append(np.zeros((1,1)))
        res = []
        for emb in self.emb_layer_model:
            res.append(emb.predict(inps))
        return res

class NgramGRUBeginFirstModel(VectorModel): #fungoval rpedtym dobre, otestovat znova

    def __init__(self, vector_size=None,negative_sample_num=5,voc_size=None,learning_rate=None,ngram_size=None,max_ngram_num=ngram_num,model=None):
        super(NgramGRUBeginFirstModel, self).__init__(vector_size,negative_sample_num,voc_size,learning_rate,model)
        self.vector_layer_name = "reshape_1"
        self.ngram_size = ngram_size
        self.max_ngram_num = max_ngram_num

    def create_emb_layer(self):
        iw = Input(shape=(self.max_ngram_num,), dtype='int32', name="inputword3")
        emb_in = embeddings.Embedding(output_dim=self.vector_size, input_dim=self.ngram_size,init="uniform",mask_zero=True,name="input_layer")
        
        vv_iw = emb_in(iw)

        zm = ZeroMaskedEntries()
        zm.build((None,self.max_ngram_num,self.vector_size))

        zero_masked_emd = zm(vv_iw)

        gru_l = recurrent.GRU(self.vector_size,return_sequences=False)

        gru = gru_l(zero_masked_emd)

        reshaped = Reshape((1,self.vector_size))(gru)
        return ([iw],emb_in,reshaped)
    
    def create_ngrams_for_word(self,word,keys,keys2=None):
        word = "^"+word+"$"
        ngrams = []
        for i in range(0,len(word)):
            for k in range(self.min_ngram_size,self.max_ngram_size+1):
                if i > len(word)-k:
                    continue
                if word[i:i+k] in keys:
                    if keys2 is None or word[i:i+k] in keys2:
                        ngrams.append(word[i:i+k])
        return ngrams

    def create_input_for_word(self,word,keys,change_ngram=None,rand_int=None,keys2=None):
        ngrams = self.create_ngrams_for_word(word, keys,keys2)
        res = [keys[ng] for ng in ngrams[:self.max_ngram_num]]
        for i in range(self.max_ngram_num-len(res)):
            res.append(0)
        if change_ngram is not None:
            res[change_ngram] = rand_int
        return np.array(res)






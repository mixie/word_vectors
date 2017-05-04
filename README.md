# Vector-space word representation using morphology

This repository contains source code for my diploma thesis - Vector-space word representation using morphology.

## File descriptions

- `models.py` - definitions of all models described in diploma thesis
- `train.py`- script for running training of models

- `vectors_compare.py` - script for comparing trained vectors to human labeled datasets

- `substitution_alg.py` - script for substitution algorithm described in thesis
- `lime_alg.py`- script for LIME algorithm described in thesis

- `helpers.py`- helper functions used in other scripts

## Command line parameter names for models
Due to compatibility reasons, command line argument names for model classes are different from real classes names, here is conversion:

- `classic_vectors` - WordVectorModel
- `ngram_sum` - NgramSumModel
- `ngram_max` - NgramMaxModel
- `ngram_conv` - NgramConvPaddedModel
- `ngram_conv2`- NgramConvMultModel
- `ngram_conv_old` - NgramConvShorterFirstModel
- `ngram_conv_old_sorted` - NgramConvBeginFirstModel
- `ngram_conv_gated` - NgramConvShorterFirstGatedModel
- `ngram_conv_old_valid` -NgramConvShorterFirstValidModel
- `ngram_conv_old_valid_whole`- NgramConvBeginFirstValidModel

    if name=="ngram_conv_old_valid_whole_gated":
        my_model = models.NgramConvBeginFirstValidGatedModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_gru2":
        my_model = models.NgramGRUMultModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)

    if name=="ngram_gru3":
        my_model = models.NgramGRUBeginFirstModel(VECTOR_SIZE,NEGATIVE_SAMPLE_NUM,voc_size,learning_rate,ngram_voc_size,max_ngram_num,model=model)



#!/usr/bin/env python
import os
import sys
# remove all std output including Using Tensorflow as backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import argparse 
import pickle
import numpy as np
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():
    # Parse input arguments and set mandatory title and description
    parser = argparse.ArgumentParser(description='Movie genre classifier from description and title')
    parser.add_argument('--title',required=True)
    parser.add_argument('--description',required=True)
    args = parser.parse_args()
    
    # Check that title and description are not empty strings
    if len(args.title)< 1 or len(args.description)<1:
        parser.print_help()
        print('title and description should not be empty strings')
        exit(-1)
    else:
        title = args.title
        desc = args.description
        predict_genre(title,desc)


def load_models(model_path = os.path.dirname(os.path.abspath(__file__))):
    
    # get model path
    model_path = os.path.join(model_path, "model")
    
    # load keras model and model parameters
    model = load_model(os.path.join(model_path, "model.h5"))
    with open(os.path.join(model_path,'model_params.pkl'), "rb") as output_file:
         model_params = pickle.load(output_file)
    
    # load tokenizers for title and description
    with open(os.path.join(model_path, 'title_tokenizer.pkl'), "rb") as output_file:
         title_tokenizer = pickle.load(output_file)
    with open(os.path.join(model_path, 'desc_tokenizer.pkl'), "rb") as output_file:
         desc_tokenizer = pickle.load(output_file)
    
    # load one-hot binary encoder
    with open(os.path.join(model_path, 'encoder.pkl'), "rb") as output_file:
         encoder = pickle.load(output_file)
    
    return model, model_params, encoder, title_tokenizer, desc_tokenizer

def generate_input(title, desc, t_tokenizer, d_tokenizer, params):
    
    # get tokenizer parameters
    maxlen_desc = params['MAX_LEN_DESC']
    maxlen_title = params['MAX_LEN_TITLE']
    
    # tokenize title and description
    desc_tokens = d_tokenizer.texts_to_sequences([desc])
    title_tokens = t_tokenizer.texts_to_sequences([title])
    
    # pad to fixed-length sequences
    desc_padded = pad_sequences(desc_tokens, padding='post', maxlen = maxlen_desc)
    title_padded = pad_sequences(title_tokens, padding='post', maxlen = maxlen_title)
    
    # horizontal stack title + description
    model_input = np.hstack((title_padded,desc_padded))
    
    return model_input
    
    
def predict_genre(title,desc):
    
    ######## LOAD MODELS ##########
    # - load the model, parameters, binary encoder and tokenizers 
    model, params, encoder, t_tokenizer, d_tokenizer = load_models()
    
    ######## PREPARE THE MODEL INPUT ##########
    # - tokenize title and description
    # - pad to fixed-length sequence
    # - put title and description together
    model_input = generate_input(title, desc, t_tokenizer, d_tokenizer, params)
    
    ######## PREDICT THE GENRE ##########
    # - use model to predict the genre
    # - convert softmax output to a string output
    # - generate and print response as json
    pred = model.predict(model_input)
    genre = encoder.inverse_transform(pred)[0]
    response = {
        'title':title,
        'description':desc,
        'genre':genre
        }
    print(json.dumps(response))
    

if __name__ == '__main__':
    main()

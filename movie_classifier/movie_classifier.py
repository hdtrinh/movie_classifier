#!/usr/bin/env python

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import argparse 
import pandas as pd
import ast
import pickle
import numpy as np
import re
import json

from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling1D, GlobalMaxPool1D,Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():
    parser = argparse.ArgumentParser(description='Movie genre classifier from description and title')
    parser.add_argument('--title',required=True)
    parser.add_argument('--description',required=True)
    args = parser.parse_args()
    if len(args.title)< 1 or len(args.description)<1:
    	parser.print_help()
    	exit(-1)
    else:
        title = args.title
        desc = args.description
        predict_genre(title,desc)


def load_models(model_path = 'movie_classifier/model/'):
    
    this_dir, this_filename = os.path.split(__file__)
    #this_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(this_dir, "model")
    
    print('model_dir',this_dir)
    
    model = load_model(os.path.join(model_path, "model.h5"))
    
    with open(os.path.join(model_path,'title_tokenizer.pkl'), "rb") as output_file:
         model_params = pickle.load(output_file)
            
    with open(os.path.join(model_path, 'title_tokenizer.pkl'), "rb") as output_file:
         title_tokenizer = pickle.load(output_file)
    with open(os.path.join(model_path, 'desc_tokenizer.pkl'), "rb") as output_file:
         desc_tokenizer = pickle.load(output_file)
    with open(os.path.join(model_path, 'encoder.pkl'), "rb") as output_file:
         encoder = pickle.load(output_file)
    
    return model, model_params, encoder, title_tokenizer, desc_tokenizer

def generate_input(title,desc,t_tokenizer, d_tokenizer):
    
    
    m, params, e, t_tokenizer, d_tokenizer = load_models()
    
    #maxlen_desc = params['maxlen_desc']
    #maxlen_title = params['maxlen_title']
    
    maxlen_desc = 300
    maxlen_title = 50
    
    desc_tokens = d_tokenizer.texts_to_sequences([desc])
    desc_padded = pad_sequences(desc_tokens, padding='post', maxlen = maxlen_desc)
    title_tokens = t_tokenizer.texts_to_sequences([title])
    title_padded = pad_sequences(title_tokens, padding='post', maxlen = maxlen_title)
    
    model_input = np.hstack((title_padded,desc_padded))
    
    return model, model_input, e
    
    
def predict_genre(title,desc):
    
    ######## LOAD MODELS ##########
    model, params, encoder, t_tokenizer, d_tokenizer = load_models()
    
    ######## PREPARE THE MODEL INPUT ##########
    
    #maxlen_desc = params['maxlen_desc']
    #maxlen_title = params['maxlen_title']
    
    maxlen_desc = 300
    maxlen_title = 50
    
    desc_tokens = d_tokenizer.texts_to_sequences([desc])
    desc_padded = pad_sequences(desc_tokens, padding='post', maxlen = maxlen_desc)
    title_tokens = t_tokenizer.texts_to_sequences([title])
    title_padded = pad_sequences(title_tokens, padding='post', maxlen = maxlen_title)
    model_input = np.hstack((title_padded,desc_padded))
    
    ######## PREDICT THE GENRE ##########
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

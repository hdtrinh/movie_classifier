
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import argparse 
import sys


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


def predict_genre(title,desc):
    model = load_model('movie_classifier/model/model.h5')
    with open('movie_classifier/model/title_tokenizer.pkl', "rb") as output_file:
         title_tokenizer = pickle.load(output_file)
    with open('movie_classifier/model/desc_tokenizer.pkl', "rb") as output_file:
         desc_tokenizer = pickle.load(output_file)
    with open('movie_classifier/model/encoder.pkl', "rb") as output_file:
         enc = pickle.load(output_file)

    max_len_desc = 300
    max_len_title = 50

    desc_tokens = desc_tokenizer.texts_to_sequences([desc])
    desc_padded = pad_sequences(desc_tokens, padding='post', maxlen = max_len_desc)
    title_tokens = title_tokenizer.texts_to_sequences([title])
    title_padded = pad_sequences(title_tokens, padding='post', maxlen = max_len_title)
    model_input = np.hstack((title_padded,desc_padded))
    pred = model.predict(model_input)
    genre = enc.inverse_transform(pred)[0]

    response = {
		'title':title,
		'description':desc,
		'genre':genre
		}

    print(json.dumps(response))

if __name__ == '__main__':
    main()

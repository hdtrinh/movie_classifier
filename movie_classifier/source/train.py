import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocessor
from encode import encoder
from model import model_classifier


def main():
    # load dataset
    df = pd.read_csv('movies_metadata.csv')
    df = df.loc[:,['title','genres','overview']]
    df = df[pd.notnull(df.overview)]
    df = df[pd.notnull(df.title)]
    df = df[pd.notnull(df.genres)]
    
    # Training parameters
    max_len_desc = 300
    max_len_title = 50
    max_input_len = max_len_title + max_len_desc
    genres_to_be_predicted = ['Drama','Comedy','Documentary','Science Fiction','Romance']
    num_classes = len(genres_to_be_predicted) 
    
    params = {
                'GENRES':           genres_to_be_predicted,
                'VOCABULARY_SIZE':  20000,
                'EMBEDDING_DIM':    100,
                'MAX_LEN_DESC':     max_len_desc,
                'MAX_LEN_TITLE':    max_len_title,
                'INPUT_LEN':        max_input_len,
                'NUM_DENSE_1':      512,
                'NUM_CLASSES':      num_classes,
                'NUM_EPOCHS':       4,
                'BATCH_DIM':        64
    
    }   
    
    # init custom classes
    p = preprocessor(genres = params['GENRES'])
    e = encoder(max_words = params['VOCABULARY_SIZE'],
                maxlen_desc = params['MAX_LEN_DESC'],
                maxlen_title = params['MAX_LEN_TITLE'])
    m = model_classifier()

    # prepare data for training
    df = p.preprocess(df)
    X, y = e.encode(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1000)
    e.save()

    # create and train model
    model = m.define_model(params)
    history = m.train_model(X_train, X_test, y_train, y_test)

    # save
    m.save_model()
    m.save_params()


if __name__ == '__main__':
    main()

import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class encoder():
    def __init__(self, model_path = '../model/', max_words = 20000,
                    maxlen_desc = 300, maxlen_title = 50):
        self.model_path = model_path
        self.max_words = max_words
        self.maxlen_title = maxlen_title
        self.maxlen_desc = maxlen_desc
        
        self.encoder = LabelBinarizer()
        self.title_tokenizer = Tokenizer(num_words= max_words,char_level=False)
        self.desc_tokenizer = Tokenizer(num_words = max_words,char_level=False)
    
    def encode(self,df):
        # One-Hot Encoding of Labels
        y = self.encoder.fit_transform(df.genre)
        
        # Tokenize the description
        self.desc_tokenizer.fit_on_texts(df.overview)
        vocabulary_size = len(self.desc_tokenizer.word_index) + 1
        X_desc = self.desc_tokenizer.texts_to_sequences(df.overview)
        
        # Tokenize the description
        self.title_tokenizer.fit_on_texts(df.title)
        X_title = self.title_tokenizer.texts_to_sequences(df.title)
        
        # Pad the sequences
        X_desc = pad_sequences(X_desc, padding = 'post', maxlen=self.maxlen_desc)
        X_title = pad_sequences(X_title, padding = 'post', maxlen=self.maxlen_title)
        
        # Form a unique input: title + description
        X = np.hstack((X_title, X_desc))
        
        return X,y
    
    def save(self):
        with open(self.model_path + 'title_tokenizer.pkl', 'wb') as outfile:
            pickle.dump(self.desc_tokenizer, outfile,  protocol=2)
    
        with open(self.model_path + 'desc_tokenizer.pkl', 'wb') as outfile:
            pickle.dump(self.title_tokenizer, outfile,  protocol=2)
    
        with open(self.model_path + 'encoder.pkl', 'wb') as outfile:
            pickle.dump(self.encoder, outfile, protocol=2)
from nltk.corpus import stopwords
import nltk
import string
import pandas as pd
from ast import literal_eval

class preprocessor():
    def __init__(self,genres=['Drama','Comedy','Documentary','Science Fiction','Romance']):
    
        # Get stopwords from NLTK
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.genres = genres

    def clean_text(self,text):    
    
        # Lower case the text and remove punctuation and stopwords
        text = text.split()
        text = " ".join(word for word in text if word not in string.punctuation)
        text = text.lower().split()    
        text = [w for w in text if not w in self.stop_words and len(w) >= 3]
        text = " ".join(text)
        return text
    
    def genre_from_list(self,list_of_genres):
    
        # Get only one genre from the list of genres
        genre = ''
        if len(list_of_genres)>0:
            genre = list_of_genres[0]['name']
        return genre
    
    def get_genre(self,df):
    
        # Get only one genre from the list of genres
        s = df.genres.apply(literal_eval).apply(self.genre_from_list)
        s.name = 'genre'
        df = df.drop('genres',axis = 1).join(s)
        return df
            
    def preprocess(self, df):
        
        # Get the genre
        # Clean title and description
        # Select the genres from genre list
        df = self.get_genre(df)
        df['overview'] = df.overview.apply(self.clean_text)
        df['title'] = df.title.apply(self.clean_text)
        df = df[df.genre.isin(self.genres)]
        return df

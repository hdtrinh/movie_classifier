import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import preprocessor
from encode import encoder
from model import model

# load dataset
df = pd.read_csv('../../../movie-genre/movies_metadata.csv')
df = df.loc[:,['title','genres','overview']]
df = df[pd.notnull(df.overview)]
df = df[pd.notnull(df.title)]
df = df[pd.notnull(df.genres)]

# init custom classes
p = preprocessor()
e = encoder()
m = model()

# prepare data for training
df = p.preprocess(df)
X, y = e.encode(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1000)
e.save()

# create and train model
model = m.define_model()
history = m.train_model(X_train, X_test, y_train, y_test)

# save
m.save_model()
m.save_params()




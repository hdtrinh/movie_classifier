# Movie Genre Classifier

A simple command-line application: given a title and a short movie description it returns an appropriate genre. 


## Install

- To test the app I suggest to create a virtual environment. 

- Clone the repository and install the app with the following commands

```
git clone https://github.com/hdtrinh/movie_classifier.git
pip3 install movie_classifier/.
```
## Usage

### Input Arguments

```
movie_classifier --title <title> --description <description>
```

Input constraints:

```
--title: the movie title. A mandatory non-empty string.
--description: the movie description. A mandatory non-empty string.
```

###  Example

```
movie_classifier --title "Othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."
```

### Output

- The model will return one of the six following genre:
- 'Drama', 'Comedy', 'Documentary', 'Science Fiction', 'Romance'

```
{ "title": "Othello", 
  "description": "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.", 
  "genre": "Drama"}
```
- In case of error please check err.txt (stderr redirected)

## Model Training

- To train the model move to source folder.
- The model uses keras and tensorflow as backend. Pandas, sklearn and numpy are used for preprocessing purposes. 
- The model gets title + description as input and return the genre as output. 


### Prerequisites

- Download the MovieLens movies_metadata.csv from here [link](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)
- Put movies_metadata.csv in movie_classifier/source folder 

### Install Training Requirements

- Move to the source folder. Then install requirements and run the training as follows. 
```
pip3 install -r requirements-train.txt
python3 train.py
```

## Training Steps
The training requires few steps defined by the following files:
- Preprocess and clean the text: preprocess.py
- One-hot encode the output and tokenize the text: encode.py
- Define the classifier model: model.py

### Preprocess

- Load the dataset into a pandas dataframe
- Use the columns title, genres, overview (description)
- Check that the mandatory fields are not empty
- Each movie can be associated to more than one genre. We perform only one genre output as simple case.
- To extend to many-genres output we can implement a multi-label classification (future works)

```
df = pd.read_csv('movies_metadata.csv')
df = df.loc[:,['title','genres','overview']]
df = df[pd.notnull(df.overview)]
df = df[pd.notnull(df.title)]
df = df[pd.notnull(df.genres)]
```

### Clean the text
- Download and use stopwords from Natural Language Toolkit nltk
- Lower-case the text
- Remove punctuation and stopwords

### Encode

- One-hot Encoding of the labels (genres)
- Tokenize separately title and description and get sequences
- Pad title and description to fixed-length sequences
- Stack together title + description

```
# One-Hot Encoding of Labels
y = self.encoder.fit_transform(df.genre)

# Tokenize the description
self.desc_tokenizer.fit_on_texts(df.overview)
X_desc = self.desc_tokenizer.texts_to_sequences(df.overview)

# Tokenize the title
self.title_tokenizer.fit_on_texts(df.title)
X_title = self.title_tokenizer.texts_to_sequences(df.title)

# Pad the sequences
X_desc = pad_sequences(X_desc, padding = 'post', maxlen=self.maxlen_desc)
X_title = pad_sequences(X_title, padding = 'post', maxlen=self.maxlen_title)

# Form a unique input: title + description
X = np.hstack((X_title, X_desc))
```

### Model 

- Define a text-classification model using Bidirectional LSTM  [[1](https://arxiv.org/pdf/1611.06639.pdf)]
- Use 1-D Global Max Pooling to gather information from temporal sequences
- Use Dense with softmax activation function to output probability scores
- The model is compiled using adam optimization and categorical cross-entropy loss
```
model = Sequential()
model.add(Embedding(input_dim=self.vocabulary_size, 
                       output_dim=self.embedding_dim, 
                       input_length=self.input_len))
model.add(Bidirectional(LSTM(self.num_dense_1, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(self.num_classes, activation='softmax'))

model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
```

## Results
- Using 6 genres ('Drama', 'Comedy', 'Documentary', 'Science Fiction', 'Romance') we get an accuracy of 63%
- No huge improvement using LSTM instead of Dense layer
- Accuracy can be improved with preprocessing (embedding with pre-trained models, e.g. GloVe, BERT) and with model tuning (e.g. using grid CV parameters search).

## Authors

* **Hoang Duy Trinh** - *Initial work* - [hdtrinh](https://github.com/hdtrinh)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* MovieLens Database
* Bidirectional LSTM [[1](https://arxiv.org/pdf/1611.06639.pdf)]




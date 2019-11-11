# Movie Genre Classifier

A simple command-line application: given a title and a short movie description it returns an appropriate genre. 


## Install

Clone the repository

```
git clone https://github.com/hdtrinh/movie_classifier.git
```

Use the following to install the requirements and the app

```
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

The model will return one of the following genre: 'Drama','Comedy','Documentary','Science Fiction','Romance'

```
{ "title": "Othello", 
  "description": "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.", 
  "genre": "Drama"}
```

## Model Training

### Prerequisites

- Download the MovieLens movies_metadata.csv from here [link](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)
- Put movies_metadata.csv in movie_classifier/source folder 

### Install Training

Move to the source folder. Then install requirements and run the training as follows:
```
pip3 install -r requirements-train.txt
python3 train.py
```

## Training Steps

### Preprocess

- Load the dataset in a pandas dataframe
- Use the columns title, genres, overview (description)
- Check that our mandatory fields are not empty
- Each movie can be associated to more than one genre. We choose just one genre output as simple case.
- To extend to many-genres output we can perform a multi-label classification (future works)

```
df = pd.read_csv('movies_metadata.csv')
df = df.loc[:,['title','genres','overview']]
df = df[pd.notnull(df.overview)]
df = df[pd.notnull(df.title)]
df = df[pd.notnull(df.genres)]
```

### Clean the text
- Use nltk stopwords
- Lower-case the text
- Remove punctuation and stopwords


### Encode

- One-hot Encoding of the labels (genres)
- Tokenizing separately title and description and get sequences
- Pad title and description to fixed-length sequences
- Stack together title + description

```
# One-Hot Encoding of Labels
y = self.encoder.fit_transform(df.genre)

# Tokenize the description
self.desc_tokenizer.fit_on_texts(df.overview)
vocabulary_size = len(self.desc_tokenizer.word_index) + 1
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

```
code
```

## Results


## Authors

* **Hoang Duy Trinh** - *Initial work* - [hdtrinh](https://github.com/hdtrinh)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* MovieLens Database
* Bidirectional LSTM [[1](https://arxiv.org/pdf/1611.06639.pdf)]




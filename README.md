# Movie Genre Classifier

A simple command-line application: given a title and a short movie description it returns an appropriate genre. 


## Install

Clone the repository

```
git clone https://github.com/hdtrinh/movie_classifier.git
```

Use the following to install the requirements and the app

```
pip3 install -r movie_classifier/requirements.txt
pip3 install movie_classifier/.
```

## Usage

### Input Arguments

```
movie_classifier --title <title> --description <description>
```

###  Example

```
movie_classifier --title "Othello" --description "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic."
```

### Output

```
{ "title": "Othello", 
  "description": "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.", 
  "genre": "Drama"}
```

## Model Training

### Preprocess

Add additional notes

### Encode

Add additional notes

### Model 

Add additional notes


## Authors

* **Hoang Duy Trinh** - *Initial work* - [hdtrinh](https://github.com/hdtrinh)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* IMDB Movie Database



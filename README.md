# Project Title

Embedding of Enron Email data using Word2Vec, FastText and GloVe

## Getting Started

The script is run on Python 3.6 environment

### Prerequisites

Required libraries

```
pip3 install nltk
pip3 install --upgrade gensim
pip3 install fasttext
pip3 install glove_python
```

## Deployment

### Eron.py 
This file is use for reading of the email, process and train the respective model.
Run this via command python3 Eron.py in terminal. Make sure to install the necessary libraries and change the path to the Enron email.csv data.

### Eron2.py
Run this file directly to run the program.
python3 Eron2.py

### Dictionary.txt
This file contain all the vocabulary involves in training Word2Vec and GloVe model. Each Vocabulary have come out at least 5 times.

### .model and .npy
Binary files generated from Eron.py


## Reference
These are the website use for reference for this assesment
```
https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
https://www.freecodecamp.org/news/how-to-get-started-with-word2vec-and-then-how-to-make-it-work-d0a2fca9dad3/
https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b
https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
https://github.com/maciejkula/glove-python/blob/master/glove/glove.py
```

## Authors
Chang Ye Han

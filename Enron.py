#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:11:48 2020

@author: yehan
"""
# =============================================================================
# Ref:
# https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
# https://www.freecodecamp.org/news/how-to-get-started-with-word2vec-and-then-how-to-make-it-work-d0a2fca9dad3/
# https://medium.com/analytics-vidhya/word-vectorization-using-glove-76919685ee0b
# https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText
# https://github.com/maciejkula/glove-python/blob/master/glove/glove.py
# =============================================================================
import numpy as np 
import pandas as pd
import email
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models import FastText
from glove import Corpus, Glove

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Path of the Enron email.csv file
filePath = '/home/yehan/Desktop/IamPlus/enron-email-dataset/emails.csv'
#readNoRows = 30000 


def get_raw_text(emails):
    email_text = []
    
    for e in emails.walk():
        if e.get_content_type() == 'text/plain':
            email_text.append(e.get_payload())
            
    return ''.join(email_text)

# Pre-processing such as tokenization, lowercasing using gensim simple_preprocess library
def readLines(sent):   
    i = 0
    for sentences in sent:
        i +=1
        if (i % 1000 == 0):
            print ("Read {0} content".format (i))
            
        yield simple_preprocess(sentences[0])

# =============================================================================
# Extracting out the content of email
# =============================================================================
#df = pd.read_csv(filePath, nrows = readNoRows)
df = pd.read_csv(filePath)

emails = list(map(email.parser.Parser().parsestr,df['message']))
headings = emails[0].keys()

for key in headings:
    df[key] = [doc[key] for doc in emails]
    
# We only interested in the content of the email, the rest can be ignore
df['body'] = list(map(get_raw_text, emails))

# Create a new data frame and focus on the body
df1 = df[['body']]

# Convert the content to a list
sent = [row.split(',') for row in df1['body']]
masterList1 = list(readLines(sent))

# Delete variable to release some memory
del df, df1, sent, emails

# =============================================================================
# Word2Vec Embedding
# =============================================================================

modelw2v = Word2Vec(masterList1, min_count=5,size = 100, workers=10, window =3, compute_loss=True)
modelw2v.get_latest_training_loss()
test = modelw2v['forecast']

modelw2v.most_similar('forecast')[:5]

# Create the dictionary.txt to store all the train vocabulary
#vocab = modelw2v.wv.vocab
#wordsInVocab = len(vocab)
#
#words = []
#for word in vocab:
#    words.append(word)
#    
#with open('Dictionary.txt', 'w') as filehandle:
#    for listitem in words:
#        filehandle.write('%s\n' % listitem)

modelw2v.save("word2vec.model")    
    

## =============================================================================
## FastText Embedding
## =============================================================================
#  
#modelft = FastText(masterList1, size=150, window=3, min_count=5, workers=10)
#
#modelft['forecast']
#modelft.most_similar('forecast')[:5]
#
#modelft.save("fasttext.model")
#
## =============================================================================
## GloVe Embedding
## =============================================================================
#
# creating a corpus object
corpus = Corpus() 

#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(masterList1, window=5)
#creating a Glove object which will use the matrix created in the above lines to create embeddings
#We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=150, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=30, no_threads=10, verbose=True)
glove.add_dictionary(corpus.dictionary)

glove.word_vectors[glove.dictionary['forecast']]

t1 = glove.most_similar('forecast')[:5]

glove.save('glove.model')

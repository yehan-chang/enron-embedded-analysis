#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:11:48 2020

@author: yehan
"""

from gensim.models import Word2Vec
from gensim.models import FastText
from glove import Glove


modelw2v = Word2Vec.load("word2vec.model")
modelft = FastText.load("fasttext.model")
modelGlove = Glove.load("glove.model")


def print_menu():       ## Your menu design here
    print (30 * "-" , "MENU" , 30 * "-")
    print ("1. Key in a word")
    print ("2. Exit")
    print (67 * "-")


def resultPage(word):
    
    print (67 * "-")    
    try:
        list1 = modelw2v.wv.most_similar(word)[:4]
        
        print ("Top 4 most similar word for Word2vec model:")
        
        for t in list1:
            print (t[0])

    except Exception as e: 
        print("The word {} not in Word2Vec vocabulary".format(word))
    print (67 * "-")    
    
    try:
        list1 = modelft.wv.most_similar(word)[:4]
        
        print ("Top 4 most similar word for FastText model:")
    
        for t in list1:
            print (t[0])

    except Exception as e: print(e) 
    print (67 * "-")    
    
    try:
        list1 = modelGlove.most_similar(word)[:4]
        
        print ("Top 4 most similar word for GloVe model:")
    
        for t in list1:
            print (t[0])

    except Exception as e: 
        print("The word {} not in GloVe dictionary".format(word))  
  
    
loop=True      
  
while loop:
    
    if __name__ == '__main__':
        print_menu()
        
    choice = input("Enter your choice [1-2]: ")
     
    if choice=='1':
        word = (input("Please key in a word: "))
        resultPage(word.lower())
        

    elif choice=='2':
        print ("Menu 2 has been selected")

        loop=False # This will make the while loop to end as not value of loop is set to False
    else:
        # Any integer inputs other than values 1-5 we print an error message
        input("Wrong option selection. Enter any key to try again..")
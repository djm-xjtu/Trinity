import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
import emoji
from googletrans import Translator
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from gensim.models import Word2Vec
from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    reviews = pd.read_csv('data/reviews.csv')
    listings = pd.read_csv('data/listings.csv')
    translator = Translator()
    a = 1
    for i, row in reviews.iterrows():
        try:
            if detect(row['comments']) != 'en':
                row['comments'] = translator.translate(row['comments']).text
        except:
            print("False")
    print(reviews.shape)
    print(listings.shape)
    reviews = pd.merge(reviews, listings)
    print(reviews.shape)
    corpus = reviews['comments'].values

    reviews.to_csv('data/comments.csv')
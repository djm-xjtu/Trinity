import gensim.models
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
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def get_comment_via_id(listing_id):
    data = pd.read_csv('data/reviews.csv')
    comments = []
    for i, row in data.iterrows():
        if row['listing_id'] == listing_id:
            comments.append(emoji.demojize(row['comments']))
    return comments


def get_scores_via_id(listing_id):
    data = pd.read_csv('data/listings.csv')
    scores = []
    for i, row in data.iterrows():
        if row['id'] == listing_id:
            scores.append(row['review_scores_rating'])
            scores.append(row['review_scores_accuracy'])
            scores.append(row['review_scores_cleanliness'])
            scores.append(row['review_scores_checkin'])
            scores.append(row['review_scores_communication'])
            scores.append(row['review_scores_location'])
            scores.append(row['review_scores_value'])
            break
    return scores


def get_all_comments():
    data = pd.read_csv('data/reviews.csv')
    comments = []
    for i, row in data.iterrows():
        comments.append(row['comments'])
    return comments


def get_word_vector_via_id(listing_id):
    comments = get_comment_via_id(listing_id)


def draw_first():
    reviews = pd.read_csv('data/reviews.csv')
    listings = pd.read_csv('data/listings.csv')
    print(reviews.groupby('listing_id').size())
    ax = reviews.groupby('listing_id').size().plot(kind='bar')
    ax.set(xlabel='lising_id', ylabel='nunber of comments')
    plt.show()


def compressWord(InputData):
    X = vectorizer.transform(InputData)
    data = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    word2Vec_data = pd.DataFrame()
    for i in range(data.shape[0]):
        sentence = np.zeros(300)
        for word in WordsVocab[data.iloc[i, :]]:
            if word in word2Vec.key_to_index.keys():
                sentence = sentence + word2Vec[word]
        word2Vec_data = word2Vec_data.append(pd.DataFrame([sentence]))
    return word2Vec_data


if __name__ == '__main__':
    reviews = pd.read_csv('data/reviews.csv')
    listings = pd.read_csv('data/listings.csv')
    reviews = pd.merge(reviews, listings)
    corpus = reviews['comments'].values

    vectorizer = CountVectorizer(stop_words='english')

    X = vectorizer.fit_transform(corpus.astype('U'))
    CountVectorizedData = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    CountVectorizedData['review_scores_rating'] = reviews['review_scores_rating']

    WordsVocab = CountVectorizedData.columns[:-1]
    word2Vec = gensim.models.KeyedVectors.load_word2vec_format(
        '/Users/dengjiaming/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    word2Vec_data = compressWord(corpus.astype('U'))
    print(word2Vec_data.shape)

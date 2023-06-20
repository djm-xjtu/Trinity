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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB

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
    print(reviews.shape)
    print(listings.shape)
    reviews = pd.merge(reviews, listings)
    print(reviews.shape)
    corpus = reviews['comments'].values

    vectorizer = CountVectorizer(stop_words='english')

    X = vectorizer.fit_transform(corpus.astype('U'))
    CountVectorizedData = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    CountVectorizedData['review_scores_rating'] = reviews['review_scores_rating']
    CountVectorizedData['review_scores_accuracy'] = reviews['review_scores_accuracy']
    CountVectorizedData['review_scores_cleanliness'] = reviews['review_scores_cleanliness']
    CountVectorizedData['review_scores_checkin'] = reviews['review_scores_checkin']
    CountVectorizedData['review_scores_communication'] = reviews['review_scores_communication']
    CountVectorizedData['review_scores_location'] = reviews['review_scores_location']
    CountVectorizedData['review_scores_value'] = reviews['review_scores_value']
    print(CountVectorizedData.shape)
    print(CountVectorizedData.head())
    WordsVocab = CountVectorizedData.columns[:-1]
    word2Vec = gensim.models.KeyedVectors.load_word2vec_format(
        '/Users/dengjiaming/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
    word2Vec_data = compressWord(reviews['comments'])
    print(word2Vec_data.shape)
    word2Vec_data.reset_index(inplace=True, drop=True)

    word2Vec_data['review_scores_rating'] = CountVectorizedData['review_scores_rating']
    word2Vec_data['review_scores_accuracy'] = CountVectorizedData['review_scores_accuracy']
    word2Vec_data['review_scores_cleanliness'] = CountVectorizedData['review_scores_cleanliness']
    word2Vec_data['review_scores_checkin'] = CountVectorizedData['review_scores_checkin']
    word2Vec_data['review_scores_communication'] = CountVectorizedData['review_scores_communication']
    word2Vec_data['review_scores_location'] = CountVectorizedData['review_scores_location']
    word2Vec_data['review_scores_value'] = CountVectorizedData['review_scores_value']

    X = word2Vec_data[word2Vec_data.columns[:-1]].values
    Y = word2Vec_data[word2Vec_data.columns[-1]].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=428)

    mlp_model = MLPClassifier(random_state=3, max_iter=300)
    mlp_model.fit(X_train, Y_train)
    mlp_prediction = mlp_model.predict(X_test)
    print(metrics.classification_report(Y_test, mlp_prediction))
    print(metrics.confusion_matrix(Y_test, mlp_prediction))
    F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
    print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

    gauss_model = GaussianNB()
    gauss_model.fit(X_train, Y_train)
    gauss_prediction = gauss_model.predict(X_test)
    print(metrics.classification_report(Y_test, gauss_prediction))
    print(metrics.confusion_matrix(Y_test, gauss_prediction))
    F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
    print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

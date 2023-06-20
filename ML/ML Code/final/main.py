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
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
# stop words
# emoji
# translate
# TODO listing features, wordVec2, logisticRegression, kNN

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
        if row['listing_id'] == listing_id:
            scores.append(row['review_scores_rating'])
            scores.append(row['review_scores_accuracy'])
            scores.append(row['review_scores_cleanliness'])
            scores.append(row['review_scores_checkin'])
            scores.append(row['review_scores_communication'])
            scores.append(row['review_scores_location'])
            scores.append(row['review_scores_value'])
            break
    return scores


if __name__ == '__main__':
    listings = pd.read_csv('data/listings.csv')
    ids = listings['listing_id'].unique()
    words = stopwords.words('english')
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', '<', '>', 'br/', 'in']:
        words.append(w)
    words = set(words)
    stemmer = SnowballStemmer('english')
    translator = Translator()
    for ID in ids:
        comments = get_comment_via_id(ID)
        scores = get_scores_via_id(ID)
        for i in range(len(comments)):
            try:
                if detect(comments[i]) != 'en':
                    translator = Translator()
                    comments[i] = translator.translate(comments[i]).text
                    # print("True")
            except:
                print("False")
        vectorizer = CountVectorizer(stop_words=words)
        X = vectorizer.fit_transform(comments)
        X = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        print(X.shape)
        print(X.head())
        X = X.toarray()
        Y = []
        for i in range(len(X)):
            Y.append(scores)
        Y = np.array(Y)
        X = np.array(X)
        print(X)
        print(X.shape)
        print(Y)
        print(Y.shape)
        print(vectorizer.get_feature_names())
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
        model_overall = MLPClassifier(random_state=3, max_iter=300)
        model_accuracy = MLPClassifier(random_state=3, max_iter=300)
        model_cleanliness = MLPClassifier(random_state=3, max_iter=300)
        model_communication = MLPClassifier(random_state=3, max_iter=300)
        model_value = MLPClassifier(random_state=3, max_iter=300)
        model_checkin = MLPClassifier(random_state=3, max_iter=300)
        model_location = MLPClassifier(random_state=3, max_iter=300)

        model_overall.fit(X_train, Y_train[:, 0])
        print(model_overall.score(X_test, Y_test[:, 0]))

        model_accuracy.fit(X_train, Y_train[:, 1])
        print(model_accuracy.score(X_test, Y_test[:, 1]))

        model_cleanliness.fit(X_train, Y_train[:, 2])
        print(model_cleanliness.score(X_test, Y_test[:, 2]))

        model_checkin.fit(X_train, Y_train[:, 3])
        print(model_checkin.score(X_test, Y_test[:, 3]))

        model_communication.fit(X_train, Y_train[:, 4])
        print(model_communication.score(X_test, Y_test[:, 4]))

        model_location.fit(X_train, Y_train[:, 5])
        print(model_location.score(X_test, Y_test[:, 5]))

        model_value.fit(X_train, Y_train[:, 6])
        print(model_value.score(X_test, Y_test[:, 6]))








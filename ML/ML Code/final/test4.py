import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
listings = pd.read_csv('data/listings.csv')
listings = listings.replace(np.nan, 0)
features = ['host_listings_count', 'host_total_listings_count', 'host_has_profile_pic', 'host_identity_verified', 'room_type', 'accommodates', 'bedrooms', 'beds', 'reviews_per_month', 'calculated_host_listings_count_shared_rooms', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count', 'instant_bookable', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d']
print(len(features))
listings['host_is_superhost'] = listings['host_is_superhost'].str.replace('t', '1')
listings['host_is_superhost'] = listings['host_is_superhost'].str.replace('f', '0')
listings['host_is_superhost'] = listings['host_is_superhost'].replace(np.nan, 0)

listings['host_acceptance_rate'] = listings['host_acceptance_rate'].str.replace('%', '').astype(float)
listings['host_has_profile_pic'] = listings['host_has_profile_pic'].str.replace('t', '1').replace('f', '0').astype(float)
listings['host_identity_verified'] = listings['host_identity_verified'].str.replace('t', '1').replace('f', '0').astype(float)
listings['room_type'] = listings['room_type'].str.replace('Entire home/apt', '4').replace('Private room', '3').replace('Shared room', '2').replace('Hotel room', '1').astype(float)
listings['instant_bookable'] = listings['instant_bookable'].str.replace('t','1').replace('f', 0).astype(float)

X = listings[features].astype(int)

Y_review_scores_rating = listings['review_scores_rating'].astype(float)
Y_review_scores_rating = Y_review_scores_rating * 100
Y_review_scores_rating = Y_review_scores_rating.astype(int)

Y_review_scores_accuracy = listings['review_scores_accuracy'].astype(float)
Y_review_scores_accuracy = Y_review_scores_accuracy * 100
Y_review_scores_accuracy = Y_review_scores_accuracy.astype(int)

Y_review_scores_cleanliness = listings['review_scores_cleanliness'].astype(float)
Y_review_scores_cleanliness = Y_review_scores_cleanliness * 100
Y_review_scores_cleanliness = Y_review_scores_cleanliness.astype(int)

Y_review_scores_checkin = listings['review_scores_checkin'].astype(float)
Y_review_scores_checkin = Y_review_scores_checkin * 100
Y_review_scores_checkin = Y_review_scores_checkin.astype(int)

Y_review_scores_communication = listings['review_scores_communication'].astype(float)
Y_review_scores_communication = Y_review_scores_communication * 100
Y_review_scores_communication = Y_review_scores_communication.astype(int)

Y_review_scores_location = listings['review_scores_location'].astype(float)
Y_review_scores_location = Y_review_scores_location * 100
Y_review_scores_location = Y_review_scores_location.astype(int)

Y_review_scores_value = listings['review_scores_value'].astype(float)
Y_review_scores_value = Y_review_scores_value * 100
Y_review_scores_value = Y_review_scores_value.astype(int)

Y_review_scores_rating = np.array(Y_review_scores_rating)
Y_review_scores_accuracy = np.array(Y_review_scores_accuracy)
Y_review_scores_cleanliness = np.array(Y_review_scores_cleanliness)
Y_review_scores_checkin = np.array(Y_review_scores_checkin)
Y_review_scores_communication = np.array(Y_review_scores_communication)
Y_review_scores_location = np.array(Y_review_scores_location)
Y_review_scores_value = np.array(Y_review_scores_value)

print('-------------review_scores_rating-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_rating, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))


print('-------------review_scores_accuracy-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_accuracy, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))


print('-------------review_scores_cleanliness-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_cleanliness, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

print('-------------review_scores_checkin-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_checkin, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

print('-------------review_scores_communication-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_communication, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

print('-------------review_scores_location-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_location, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

print('-------------review_scores_value-------------')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_review_scores_value, test_size=0.3, random_state=40)
mlp_model = MLPClassifier(random_state=3, max_iter=300)
mlp_model.fit(X_train, Y_train)
mlp_prediction = mlp_model.predict(X_test)
print(mlp_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, mlp_prediction))
F1_Score = metrics.f1_score(Y_test, mlp_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))

gauss_model = GaussianNB()
gauss_model.fit(X_train, Y_train)
gauss_prediction = gauss_model.predict(X_test)
print(gauss_model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, gauss_prediction))
F1_Score = metrics.f1_score(Y_test, gauss_prediction, average='weighted')
print('F1 score of the model on Testing Sample Data:', round(F1_Score, 2))
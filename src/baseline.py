import pandas as pd
import numpy as np
from src.preprocessing import Preprocessing
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class Baseline(object):
    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def transform_baseline(self,x_train,x_test,y_train,y_test):
        X = x_train.append(x_test)
        y = y_train.append(y_test)
        wnl = WordNetLemmatizer()
        stem = SnowballStemmer('english')
        for row in X:
            row = wnl.lemmatize(row)
            row = stem.stem(row)
        bagofwords_vectorizer = CountVectorizer(stop_words='english')
        bagofwords = bagofwords_vectorizer.fit_transform(X).toarray()
        x_train = bagofwords[:len(x_train)]
        x_test = bagofwords[len(x_train):]
        label = LabelEncoder()
        y = label.fit_transform(y)
        y_train = y[:len(y_train)]
        y_test = y[len(y_train):]
        return x_train, x_test, y_train, y_test

    def run_baseline(self,x_train,x_test,y_train,y_test):
        nb = MultinomialNB()
        nb.fit(x_train,y_train)
        print(accuracy_score(y_test,nb.predict(x_test)))

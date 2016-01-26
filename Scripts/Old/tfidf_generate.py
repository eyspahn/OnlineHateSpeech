# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer


def splitdata(X, y, classes, test_size=0.3):
    '''
    Split data into test & train; binarizes y into 5 classes
    Outputs: X_train, X_test, y_train, y_test
    '''

    return train_test_split(X, y, test_size=test_size, random_state=42)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


if __name__ == '__main__':
    # main()
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']
    stemmer = SnowballStemmer("english")

    print("Loading Data")
    X_stripped = pickle.load(open('X_stripped.p', 'rb'))
    y = pickle.load(open('y.p', 'rb'))

    print("Splitting Data")
    X_train, X_test, y_train, y_test = splitdata(X_stripped, y, classes)

    tfidfv = TfidfVectorizer(decode_error='ignore', stop_words='english',
                            tokenizer=tokenize)
    tfidf_X_train = tfidfv.fit_transform(X_train)
    tfidf_X_test = tfidfv.transform(X_test)

    pickle.dump(tfidfv, open('tfidfv.p', 'wb'))
    pickle.dump(tfidf_X_train, open('tfidf_X_train.p', 'wb'))
    pickle.dump(tfidf_X_test, open('tfidf_X_test.p', 'wb'))

# -*- coding: utf-8 -*-
'''Run a prediction for a comment through the reddit May 2015 hate speech model'''

from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball
import xgboost as xgb
import cPickle as pickle
import numpy as np
import pandas as pd


stemmer = snowball.SnowballStemmer("english")


def stem_tokens(tokens, stemmer):
    '''Stem the tokens.'''
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    '''Tokenize & stem. Stems automatically for now.
    Leaving "stemmer" out of function call, so it works with TfidfVectorizer'''
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def predict_comment(comment, classes, bst, vect):
    '''
    Where "comment" is the comment by the user, to be passed in.
    classes =
    '''
    comment_tfidf = vect.transform([comment])
    comment_xgb = xgb.DMatrix(comment_tfidf)
    yprob = bst.predict(comment_xgb).reshape(1, 5)  # hard coding -- only one comment at a time in this case.
    ylabel = classes[np.argmax(yprob, axis=1)]

    # print('The class is: {0} with probability {1}%'.format(ylabel, round(100 * np.max(yprob), 1)))

    return ylabel, round(100*np.max(yprob), 1), comment


def main():
    classes = ['Not Hate', 'Size Hate', 'Gender Hate', 'Race Hate', 'Religion Hate']

    # load saved xgboost model
    bst = xgb.Booster()
    bst.load_model('../FinalModel/modelv1/BuildModel/hatespeech.model')
    # load tf-idf matrix
    # tfidf_X = pickle.load(open('../FinalModel/BuildModel/tfidf_X.p', 'rb'))
    vect = pickle.load(open('../FinalModel/modelv1/BuildModel/vect.p', 'rb'))

    # get comment from user
    comment = raw_input('Enter comment: ')
    # predict class of comment
    predict_comment(comment, classes, bst, vect)

    predict = raw_input("Enter 'y' to get another prediction.")

    while predict == 'y':
        # get comment from user
        comment = raw_input('Enter comment: ')
        # predict class of comment
        predict_comment(comment, classes, bst, vect)
        predict = raw_input("Enter 'y' to get another prediction.")


if __name__ == "__main__":
    main()

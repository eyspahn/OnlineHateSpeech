# -*- coding: utf-8 -*-

import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from nltk import word_tokenize
from nltk.stem import snowball


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename='../Data/labeledhate_5cats.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


def splitdata(df, test_size=0.3):
    '''
    Split data into test & train; binarizes y into 5 classes
    Outputs: X_train, X_test, y_train, y_test
    '''
    X = df.body
    y = df.label
    ybin = label_binarize(y, classes=['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate'])
    return train_test_split(X, ybin, test_size=test_size, random_state=42)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def vectorizer(vectchoice='Count', stopwords='english', tokenize_me=None, max_features=500):
    '''
    Choose/return sklearn vectorizer, from Count Vectorizer, TFIDF, HashingVectorizer
    Choose from: stopwords: ['english' or 'None'],
                vectorizer = [Count, Hash or Tfidf]
                tokenize_me: [None or tokenize]

    '''

    if vectchoice == 'Count':
        vect = CountVectorizer(stop_words=stopwords, decode_error='ignore',
                            tokenizer=tokenize_me, max_features=max_features)

    # class sklearn.feature_extraction.text.CountVectorizer(input='content', encoding='utf-8',
    # decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
    # stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word',
    # max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
    # dtype=<class 'numpy.int64'>)[source]¶

    elif vectchoice == "Hash":
        vect = HashingVectorizer(stop_words=stopwords, decode_error='ignore',
                                non_negative=True, tokenizer=tokenize_me)

    # class sklearn.feature_extraction.text.HashingVectorizer(input='content', encoding='utf-8',
    # decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
    #  stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word',
    #  n_features=1048576, binary=False, norm='l2', non_negative=False, dtype=<class 'numpy.float64'>)[source]

    elif vectchoice == 'Tfidf':
        vect = TfidfVectorizer(stop_words=stopwords, decode_error='ignore',
                                tokenizer=tokenize_me, max_features=max_features)

    # class sklearn.feature_extraction.text.TfidfVectorizer(input='content', encoding='utf-8',
    # decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
    # analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1),
    # max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>
    # , norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)[source]¶

    # fit & transform train vector
    vectfit_train = vect.fit_transform(X_train)
    # transform test vector
    vectfit_test = vect.transform(X_test)

    return vectfit_train, vectfit_test


def print_scores(y_test, y_score, y_preds):
    '''
    Print model performance stats.
    y_test: test labels; y_score: probabilities; y_preds: predictions
    '''
    print("ROC AUC Score: {0}".format(roc_auc_score(y_test, y_score)))
    print(classification_report(y_test, y_preds))


def createmulticlassROC(classes, y_test, y_score):
    '''
    Function to create & plot ROC curve & associated areas
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Inputs: n_classes: the number of classes
            y_test: the test labels
            y_score: the predicted probabilities for each class.
                (e.g. y_score = classifier.fit(countv_fit_X_train, y_train).predict_proba(countv_fit_X_test) )
    '''
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12, 8))

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for multi-class')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # main()
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']

    print('Loading Data')
    df = load_data()
    print('Splitting Data')
    X_train, X_test, y_train, y_test = splitdata(df)

    # loop through options & print out scores
    # vect_options = ['Count', 'Hash', 'Tfidf']
    stemmer_options = [snowball.SnowballStemmer("english")]
    # stemmer_options = [snowball.SnowballStemmer("english"), porter.PorterStemmer()]
    # token_options = [None, tokenize]
    vect_options = ['Tfidf']
    # stemmer_options = [snowball.SnowballStemmer("english")]
    token_options = [tokenize]

    for token in token_options:
        for stemmer in stemmer_options:
            for vect in vect_options:
                print('For vect {0}, stemmer {1} & token {2}'.format(vect, stemmer, token))
                vectfit_X_train, vectfit_X_test = vectorizer(vectchoice=vect, tokenize_me=token,
                                stopwords='english')
                print('Classifying')
                classifier = OneVsRestClassifier(MultinomialNB(), n_jobs=1)
                fitted_clf = classifier.fit(vectfit_X_train, y_train)
                print('Predicting')
                y_score = fitted_clf.predict_proba(vectfit_X_test)
                y_preds = fitted_clf.predict(vectfit_X_test)

                # createmulticlassROC(classes, y_test, y_score)
                # plt.savefig('MultinomialNB, CountV, Not Tokenized')
                print_scores(y_test, y_score, y_preds)

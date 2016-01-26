# -*- coding: utf-8 -*-
'''Run cross validation on tf-idf + xgboost model'''

# import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, auc
# from sklearn.grid_search import GridSearchCV

import xgboost as xgb
import cPickle as pickle

from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

stemmer = snowball.SnowballStemmer("english")


def load_data(filename='../Data/labeledhate_5cats.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


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




def print_scores(y_testbin, y_score, y_preds):
    '''
    Print model performance stats.
    y_testbin: test labels (binned); y_score: probabilities; y_preds: predictions
    '''
    print("ROC AUC Score: {0}".format(roc_auc_score(y_testbin, y_score)))
    # print(classification_report(y_testbin, y_preds))


def main():
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']
    print('Loading Data')
    df = load_data()
    X = df.body
    y = df.label

    # relabel y labels to reflect integers [0-4] for xgboost
    for ind in range(len(classes)):
        y[(y == classes[ind])] = int(ind)

    # binarize labels for sklearn--NOT USED for xgboost!
    # For xgboost, use single column array of values 0, 1, 2, 3, 4
    # ybin = label_binarize(y, classes=['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate'])

    kfoldcount = 0
    kf = KFold(y.shape[0], n_folds=5, shuffle=True)

    for train_index, test_index in kf:
        kfoldcount += 1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # relabel the output for multiclass roc plot, score
        y_testbin = label_binarize(y_test.astype(int),
                                   classes=[0, 1, 2, 3, 4], sparse_output=False)

        print("Vectorizing")
        vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
                               tokenizer=tokenize)

        # fit & transform train vector
        tfidf_X_train = vect.fit_transform(X_train)
        # transform test vector
        tfidf_X_test = vect.transform(X_test)

        # develop train data
        xg_train = xgb.DMatrix(tfidf_X_train, label=y_train)
        # develop test data
        xg_test = xgb.DMatrix(tfidf_X_test, label=y_test)

        print('Classifying')

        # Set up xboost parameters
        # use softmax multi-class classification to return probabilities
        param = {'objective': 'multi:softprob',
                 'eta': 0.9,
                 'max_depth': 6,
                 'num_class': 5,
                 }

        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        num_round = 500
        evals_result_dict = {}
        bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=5,
                        evals_result=evals_result_dict)

        print('Predicting')
        # get prediction, this is in 1D array, need reshape to (ndata, nclass)
        # probabilities
        y_proba = bst.predict(xg_test).reshape(y_test.shape[0], 5)
        # predictions
        y_preds = np.argmax(y_proba, axis=1)

        # pdb.set_trace()

        createmulticlassROC(classes, y_testbin, y_proba)
        plt.savefig('ROCcurves_{0}.png'.format(kfoldcount))
        # with open('model_run_info_{0}.txt'.format(kfoldcount), 'w') as f:
        #     f.write(print_scores(y_testbin, y_proba, y_preds))
        #     # f.write('evals_result')
        #     f.write('\n')
        #     f.write(confusion_matrix(y_testbin, y_preds))

if __name__ == '__main__':
    main()

'''Run xgboost on AWS, using saved out tfidf files'''

# -*- coding: utf-8 -*-

import cPickle as pickle
import xgboost as xgb

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, auc

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def splitdata(X, y, classes, test_size=0.3):
    '''
    Split data into test & train; binarizes y into 5 classes
    Outputs: X_train, X_test, y_train, y_test
    '''

    # relabel y labels to reflect integers [0-4] for xgboost
    for ind in range(len(classes)):
        y[(y == classes[ind])] = int(ind)

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


def print_scores(y_test, y_score, y_preds):
    '''
    Print model performance stats.
    y_test: test labels (sklearn format); y_score: probabilities; y_preds: predictions
    '''
    try:
        print("ROC AUC Score: {0}".format(roc_auc_score(y_test, y_score)))
        # print("Classification report: {0}".format(classification_report(y_test, y_preds)))
    except:
        pass


def top_features(d, n=20):
    '''
    Function to show the top n important features & their scores.
    The get_fscore method in xgboost returns a dictionary of features & a number.
    Get the top n features with the highest scores

    d is a dictionary (from get_fscore method in xgboost)
    '''

    featureslist = []

    for k, v in sorted(d.iteritems(), reverse=True, key=lambda (k, v): (v, k)):
        featureslist.append((k, v))

    return featureslist[:n]


def top_features_words(d, vect, n=20):
    '''
    Function to show the top n important features, their scores,and corresponding words.
    The get_fscore method in xgboost returns a dictionary of features & a number.
    Get the top n features with the highest scores

    d is a dictionary (from bst.get_fscore() in xgboost)
    vect is the instantiated vectorizer (e.g. vect = TfidfVectorizer(stuff); not the fitted variable name)
    '''
    try:
        # Back out important features
        dicta = vect.vocabulary_
        dictb = dict((v, k) for k, v in dicta.items())
        # dictb[featurenum] returns the word.

        featureslist = []
        for k, v in sorted(d.iteritems(), reverse=True, key=lambda (k, v): (v, k)):
            featureslist.append((k, v))

        topfeatures = []
        for i in xrange(n):
            fname = featureslist[i][0]
            fnum = int(filter(lambda x: x.isdigit(), fname))
            topfeatures.append((featureslist[i][0], featureslist[i][1], dictb[fnum]))

        return topfeatures

    except:
        pass


def createmulticlassROC(classes, y_test, y_score):
    '''
    Function to create & plot ROC curve & associated areas
    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    Inputs: classes: a list of classes
            y_test: the test labels, binarized into columns
            y_score: the predicted probabilities for each class.
                (e.g. y_score = classifier.fit(countv_fit_X_train, y_train).predict_proba(countv_fit_X_test) )
    '''

    # Compute ROC curve and ROC area for each class
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    # plt.figure(figsize = (12,8))
    plt.figure()

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
    # plt.show()


def main():

    stemmer = SnowballStemmer("english")
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']

    print("Loading Data")
    X_stripped = pickle.load(open('../X_stripped.p', 'rb'))
    y = pickle.load(open('../y.p', 'rb'))

    print("Splitting Data")
    X_train, X_test, y_train, y_test = splitdata(X_stripped, y, classes)

    # relabel the output for multiclass roc plot, score
    ylabel_bin = label_binarize(y_test.astype(int), classes=[0, 1, 2, 3, 4], sparse_output=False)

    # load in previously generated tfidf matrices
    tfidfv = pickle.load(open('../tfidfv.p', 'rb'))
    tfidf_X_train = pickle.load(open('../tfidf_X_train.p', 'rb'))
    tfidf_X_test = pickle.load(open('../tfidf_X_test.p', 'rb'))

    xg_train = xgb.DMatrix(tfidf_X_train, label=y_train)
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

    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=5)

    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    yprob = bst.predict(xg_test).reshape(y_test.shape[0], 5)
    ylabel = np.argmax(yprob, axis=1)

    # save out model for future use
    bst.save_model('xgb.model')

    # # dump model & feature map
    # bst.dump_model('dump.raw.txt', 'featmap.txt')

    # generate the ROC curve
    createmulticlassROC(classes, ylabel_bin, yprob)
    plt.savefig('ROCcurves.png')

    # Save the top features for the model run & print out overall ROC score
    top_words = top_features_words(bst.get_fscore(), tfidfv, n=20)
    with open('model_run_info.txt', 'w') as f:
        f.write('predicting, classification error=%f' % (sum(int(ylabel[i]) != y_test.iloc[i] for i in range(len(y_test))) / float(len(y_test))))
        f.write('\n')
        f.write("ROC AUC Score: {0}".format(roc_auc_score(ylabel_bin, yprob)))
        f.write('\n')
        if type(top_words) == str:
            f.write(top_words)
        else:
            f.write('Boo, top_words returned nothing this time & you need to find out why.')


if __name__ == '__main__':
    main()

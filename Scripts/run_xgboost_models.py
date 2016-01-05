# -*- coding: utf-8 -*-

import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

import xgboost as xgb

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, auc

from nltk import word_tokenize
from nltk.stem import snowball, porter, wordnet
#SnowballStemmer, porter.PorterStemmer, wordnet.WordNetLemmatizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def load_data(filename = '../Data/labeledhate_5cats.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))

def splitdata(df, classes, test_size=0.3):
    '''
    Split data into test & train; binarizes y into 5 classes
    Outputs: X_train, X_test, y_train, y_test
    '''
    X = df.body
    y = df.label

    #relabel y labels to reflect integers [0-4] for xgboost
    for ind in range(len(classes)):
        y[(y==classes[ind])] = int(ind)

    return train_test_split(X, y, test_size=test_size, random_state=42)


def stem_tokens(tokens, stemmer):
    stemmed=[]
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def vectorizer(vectchoice = 'Count', stopwords = 'english', tokenize_me = None, max_features=5000):
    '''
    Choose/return sklearn vectorizer, from Count Vectorizer, TFIDF, HashingVectorizer
    Choose from: stopwords: ['english' or 'None'],
                vectorizer = ['Count', 'Hash' or 'Tfidf']
                tokenize_me: [None or tokenize]
    '''

    if vectchoice == 'Count':
        vect = CountVectorizer(stop_words=stopwords, decode_error='ignore',
                                tokenizer = tokenize_me, max_features=max_features)

    # class sklearn.feature_extraction.text.CountVectorizer(input='content', encoding='utf-8',
    # decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
    # stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word',
    # max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False,
    # dtype=<class 'numpy.int64'>)[source]¶



    elif vectchoice == "Hash":
        vect = HashingVectorizer(stop_words=stopwords, decode_error = 'ignore',
                                    non_negative=True, tokenizer = tokenize_me)

    # class sklearn.feature_extraction.text.HashingVectorizer(input='content', encoding='utf-8',
    # decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None,
    #  stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word',
    #  n_features=1048576, binary=False, norm='l2', non_negative=False, dtype=<class 'numpy.float64'>)[source]



    elif vectchoice == 'Tfidf':
        vect = TfidfVectorizer(stop_words=stopwords, decode_error='ignore',
                                tokenizer = tokenize_me,max_features=max_features)


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
    y_test: test labels (sklearn format); y_score: probabilities; y_preds: predictions
    '''
    print("ROC AUC Score: {0}".format(roc_auc_score(y_test, y_score)))
    # print("Classification report: {0}".format(classification_report(y_test, y_preds)))

def top_features(d, n=20):
    '''
    Function to show the top n important features & their scores.
    The get_fscore method in xgboost returns a dictionary of features & a number.
    Get the top n features with the highest scores

    d is a dictionary (from get_fscore method in xgboost)
    '''

    featureslist = []

    for k, v in sorted(d.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        featureslist.append((k,v))

    return featureslist[:n]


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
    plt.figure(figsize = (12,8))

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
    #main()
    classes = ['NotHate', 'SizeHate', 'GenderHate', 'RaceHate', 'ReligionHate']

    print("Loading Data")
    df = load_data()
    print("Splitting Data")
    X_train, X_test, y_train, y_test = splitdata(df, classes)

    #relabel the output for multiclass roc plot, score
    ylabel_bin = label_binarize(y_test.astype(int), classes=[0,1,2,3,4],sparse_output=False)

    ### Loop through # max_features? --> Use 5000 as a starting point, at least for now. ###
    ### Use english stop words
    ### Loop Through Vectorizer, stemmer/tokenizing ###
    vect_options = ['Count', 'Hash', 'Tfidf']
    # vect_options = ['Hash']

    stemmer_options = [snowball.SnowballStemmer("english"), porter.PorterStemmer()]
    #Note - wordnet.WordNetLemmatizer() has no .stem option & doesn't fit the format of this code.

    # token_options = [None, tokenize]
    token_options = [tokenize]


    for token in token_options:
        for stemmer in stemmer_options:
            for vect in vect_options:
                print('For vect {0}, stemmer {1} & token {2}'.format(vect, stemmer, token))
                print('Vectorizing')
                vectfit_X_train, vectfit_X_test = vectorizer(vectchoice = vect,
                            stopwords = 'english', tokenize_me = token)
                print('Classifying')
                xg_train = xgb.DMatrix(vectfit_X_train, label=y_train)
                xg_test = xgb.DMatrix(vectfit_X_test, label=y_test)
                # Set up xboost parameters
                param = {}
                # use softmax multi-class classification
                # param['objective'] = 'multi:softmax'
                # scale weight of positive examples
                param['eta'] = 0.9
                param['max_depth'] = 6
                param['silent'] = 1
                param['nthread'] = 4
                param['num_class'] = 5

                watchlist = [ (xg_train, 'train'), (xg_test, 'test') ]
                num_round = 5
                # bst = xgb.train(param, xg_train, num_round, watchlist )
                param['objective'] = 'multi:softprob'
                bst = xgb.train(param, xg_train, num_round, watchlist );
                # get prediction, this is in 1D array, need reshape to (ndata, nclass)
                yprob = bst.predict(xg_test).reshape(y_test.shape[0], 5)
                ylabel = np.argmax(yprob, axis=1)

                print('predicting, classification error=%f' % (sum( int(ylabel[i]) != y_test.iloc[i] for i in range(len(y_test))) / float(len(y_test)) ))
                #createmulticlassROC(classes, ylabel_bin, yprob)
                print("ROC AUC Score: {0}".format(roc_auc_score(ylabel_bin, yprob)))

                top_features(bst.get_fscore(), n=20)
                print(' ')


    # createmulticlassROC(classes, y_test, y_score)
    # plt.savefig("RandomForest_tfidf_notokens")

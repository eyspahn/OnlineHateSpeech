import cPickle as pickle
from gensim.models import Word2Vec
import re


# Prepare data to run Word2Vec
# Needs data to be fed in by sentence, which is a list of lists.
# [ [sentence, 1, words, go, here], [sentence, 2, workds, go, here]]
# http://radimrehurek.com/gensim/models/word2vec.html


def load_data(filename = '../Data/labeledhate_5cats.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))

def separate_data(df):
    '''
    Extract comments only into Hate categories, for separate preprocessing
    Word2Vec does not need labels.
    '''

    NotHateComments = df[df.label=='NotHate']
    SizeHateComments = df[df.label=='SizeHate']
    GenderHateComments = df[df.label=='GenderHate']
    ReligionHateComments = df[df.label=='ReligionHate']
    RaceHateComments = df[df.label=='RaceHate']

    return [NotHateComments.body, SizeHateComments.body, GenderHateComments.body,
            ReligionHateComments.body, RaceHateComments.body]

def comment2sentence(comment):
    '''
    Convert comments to sentences & split into list of sentences
    Use re to split on multiple endings. Returns a list of separated comments.
    NOTE: currently will not strip leading whitespace.
    '''

    return re.split('[\.\?\!]', comment)


def sentence2list(sentence):
    '''
    Split sentences on white spaces and put into a list.
    '''

    return sentence.split()


def prepare_comments(comments):
    '''
    Prepare a set of comments for reading in & processing by
    '''

    fullcommentslist = []

    #go through each comment
    for comment in comments:
        #break comments into list of sentences
        listofsent = comment2sentence(comment)
        #split each sentence into a list of words
        for sent in listofsent:
            listofwordsofsent = sentence2list(sent)
            fullcommentslist.append(listofwordsofsent)

    return fullcommentslist


if __name__ == '__main__':
    print('Loading Data')
    df = load_data()
    print('Separating Comments')
    NotHateComments, SizeHateComments, GenderHateComments, ReligionHateComments, RaceHateComments = separate_data(df)
    print('Preparing Comments')
    SizeHatePreparedComments = prepare_comments(SizeHateComments)
    print('Running SizeHateModel')
    # SizeHateModel = Word2Vec(SizeHatePreparedComments, size=200, workers=3)
    # #save model for further use
    # SizeHateModel.save('../Data/word2vecmodels/SizeHateModel.model')
    # #SizeHateModel=Word2Vec.load('../Data/word2vecmodels/SizeHateModel.model')
    NotHateModel = Word2Vec(NotHateComments, size=200, workers=3)
    NotHateModel.save('../Data/word2vecmodels/NotHateModel.model')

    GenderHateModel = Word2Vec(GenderHateComments, size=200, workers=3)
    GenderHateModel.save('../Data/word2vecmodels/GenderHateModel.model')

    ReligionHateModel = Word2Vec(ReligionHateComments, size=200, workers=3)
    ReligionHateModel.save('../Data/word2vecmodels/ReligionHateModel.model')

    RaceHateModel = Word2Vec(RaceHateComments, size=200, workers=3)
    RaceHateModel.save('../Data/word2vecmodels/RaceHateModel.model')    

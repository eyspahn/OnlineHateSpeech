import cPickle as pickle
from gensim.models import Word2Vec
import re
import string


# Prepare data to run Word2Vec
# Needs data to be fed in by sentence, which is a list of lists.
# [ [sentence, 1, words, go, here], [sentence, 2, words, go, here] ]
# http://radimrehurek.com/gensim/models/word2vec.html


def load_data(filename='../Data/labeledhate_5cats.p'):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


def separate_data(df):
    '''
    Extract comments only into Hate categories, for separate preprocessing
    Word2Vec does not need labels.
    '''
    NotHateComments = df[df.label == 'NotHate']
    SizeHateComments = df[df.label == 'SizeHate']
    GenderHateComments = df[df.label == 'GenderHate']
    ReligionHateComments = df[df.label == 'ReligionHate']
    RaceHateComments = df[df.label == 'RaceHate']
    AllComments = df.body
    return [NotHateComments.body, SizeHateComments.body, GenderHateComments.body,
            ReligionHateComments.body, RaceHateComments.body, AllComments]


def separate_data_hate_or_not(df):
    '''
    Extract comments only into Hate & Not Hate
    Word2Vec does not need labels.
    '''

    NotHateComments = df[df.label == 'NotHate']
    HateComments = df[df.label != 'NotHate']

    return [NotHateComments.body, HateComments.body]


def comment2sentence(comment):
    '''
    Convert comments to sentences & split into list of sentences
    Use re to split on multiple endings. Returns a list of separated comments.
    NOTE: currently will not strip leading whitespace.
    '''

    return re.split('[\.\?\!]', comment)


def sentence2list(sentence):
    '''
    Makes words lowercase, removes punctuation, splits words on white space & returns list.
    '''
    # make lowercase
    sentence = sentence.lower()
    # remove punctuation
    sentence = ''.join(l for l in sentence if l not in string.punctuation)

    # return split sentence in a list
    return sentence.split()


def prepare_comments(comments):
    '''
    Prepare a set of comments for reading in & processing by word2vec
    '''

    fullcommentslist = []

    # go through each comment
    for comment in comments:
        # break comments into list of sentences
        listofsent = comment2sentence(comment)
        # split each sentence into a list of words
        for sent in listofsent:
            listofwordsofsent = sentence2list(sent)
            fullcommentslist.append(listofwordsofsent)

    return fullcommentslist


if __name__ == '__main__':
    print('Loading Data')
    df = load_data()

    print('Separating Comments')
    NotHateComments, HateComments = separate_data_hate_or_not(df)

    print('Running HateModel')
    HatePreparedComments = prepare_comments(HateComments)
    HateModel = Word2Vec(HatePreparedComments, size=200, workers=3)
    HateModel.save('../Data/word2vecmodels/HateModel.model')


#     print('Separating Comments')
#     NotHateComments, SizeHateComments, GenderHateComments, ReligionHateComments,\
#     RaceHateComments, AllComments = separate_data(df)

#     print('Running SizeHateModel')
#     SizeHatePreparedComments = prepare_comments(SizeHateComments)
#     SizeHateModel = Word2Vec(SizeHatePreparedComments, size=200, workers=3)
#     SizeHateModel.save('../Data/word2vecmodels/SizeHateModel.model')
#     # SizeHateModel=Word2Vec.load('../Data/word2vecmodels/SizeHateModel.model')

#     print('Running NotHateModel')
#     NotHatePreparedComments = prepare_comments(NotHateComments)
#     NotHateModel = Word2Vec(NotHatePreparedComments, size=200, workers=3)
#     NotHateModel.save('../Data/word2vecmodels/NotHateModel.model')

#     print('Running GenderHateModel')
#     GenderHatePreparedComments = prepare_comments(GenderHateComments)
#     GenderHateModel = Word2Vec(GenderHatePreparedComments, size=200, workers=3)
#     GenderHateModel.save('../Data/word2vecmodels/GenderHateModel.model')

#     print('Running ReligionHateModel')
#     ReligionHatePreparedComments = prepare_comments(ReligionHateComments)
#     ReligionHateModel = Word2Vec(ReligionHatePreparedComments, size=200, workers=3)
#     ReligionHateModel.save('../Data/word2vecmodels/ReligionHateModel.model')

#     print('Running RaceHateModel')
#     RaceHatePreparedComments = prepare_comments(RaceHateComments)
#     RaceHateModel = Word2Vec(RaceHatePreparedComments, size=200, workers=3)
#     RaceHateModel.save('../Data/word2vecmodels/RaceHateModel.model')

#     #All comments
#     print('Running AllCommentsModel')
#     AllCommentsPreparedComments = prepare_comments(AllComments)
#     AllCommentsModel = Word2Vec(AllCommentsPreparedComments, size=300, workers=3)
#     AllCommentsModel.save('../Data/word2vecmodels/AllCommentsModel.model')

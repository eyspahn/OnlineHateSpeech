# -*- coding: utf-8 -*-
'''Drive a twitter bot to take a comment & reply with the hate speech prediction
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball
import xgboost as xgb
import cPickle as pickle
import numpy as np
import pandas as pd

import ConfigParser
import json

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API

stemmer = snowball.SnowballStemmer("english")

config = ConfigParser.ConfigParser()
config.read('twitter.cfg')
consumer_key = config.get('apikey', 'key')
consumer_secret = config.get('apikey', 'secret')
access_token = config.get('token', 'token')
access_token_secret = config.get('token', 'secret')
stream_rule = '@hatespeechbot'
account_screen_name = 'hatespeechbot'
account_user_id = '752730899611996161'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
twitterApi = API(auth)


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

# load saved xgboost model
bst = xgb.Booster()
bst.load_model('../FinalModel/modelv1/BuildModel/hatespeech.model')
# load tf-idf matrix
vect = pickle.load(open('../FinalModel/modelv1/BuildModel/vect.p', 'rb'))

def predict_comment(comment, classes, bst, vect):
    '''
    Where "comment" is the comment by the user, to be passed in.
    '''
    comment_tfidf = vect.transform([comment])
    comment_xgb = xgb.DMatrix(comment_tfidf)
    yprob = bst.predict(comment_xgb).reshape(1, 5)  # hard coding -- only one comment at a time in this case.
    ylabel = classes[np.argmax(yprob, axis=1)]

    return 'The comment is {0} with probability {1}%'.format(ylabel, round(100 * np.max(yprob), 1))


class ReplyToTweet(StreamListener):

    def on_data(self, data):
        print '----'
        print data
        tweet = json.loads(data.strip())

        retweeted = tweet.get('retweeted')
        from_self = tweet.get('user',{}).get('id_str','') == account_user_id

        if retweeted is not None and not retweeted and not from_self:

            tweetId = tweet.get('id_str')
            screenName = tweet.get('user',{}).get('screen_name')
            tweetText = tweet.get('text')

            classes = ['Not Hate', 'Size Hate', 'Gender Hate', 'Race Hate', 'Religion Hate']
            comment_prediction = predict_comment(tweetText, classes, bst, vect)

            replyText = '@' + screenName + ' ' + comment_prediction

            #check if repsonse is over 140 char
            if len(replyText) > 140:
                replyText = replyText[0:139] + '…'

            print('Tweet ID: ' + tweetId)
            print('From: ' + screenName)
            print('Tweet Text: ' + tweetText)
            print('Reply Text: ' + replyText)

            # If rate limited, the status posts should be queued up and sent on an interval
            twitterApi.update_status(status=replyText, in_reply_to_status_id=tweetId)
            # insert a time.sleep(6) for 6 second interval between tweets?

    def on_error(self, status):
        print status
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

def main():


    streamListener = ReplyToTweet()
    twitterStream = Stream(auth, streamListener)
    # twitterStream.userstream(_with='user')
    print "Ready to listen."
    twitterStream.userstream(replies=all)

if __name__ == '__main__':
    main()

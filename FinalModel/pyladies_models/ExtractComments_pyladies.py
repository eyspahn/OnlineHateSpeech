''' Go from the May2015 reddit comments sqlite database and extract & save a subset for later use
Data in 30 GB sqlite database file available from https://www.kaggle.com/reddit/reddit-comments-may-2015
'''

import sqlite3
import pandas as pd
import cPickle as pickle

# Set up connection to database
sqlite_file = '/Volumes/ja2/ja2_RedditProject/Data/database.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# Create list of relevant subreddits, both hateful and not hateful.
# not hateful subreddits
final_nothate_srs = ['politics', 'worldnews', 'history', 'blackladies', 'lgbt',
                     'TransSpace', 'women', 'TwoXChromosomes', 'DebateReligion',
                     'religion', 'islam', 'Judaism', 'BodyAcceptance', 'fatlogic']
# List of hateful subreddits - note, removed KotakuInAction from the list, was originally included
final_hateful_srs = ['CoonTown', 'WhiteRights', 'Trans_fags', 'SlutJustice',
                     'TheRedPill', 'IslamUnveiled', 'GasTheKikes',
                     'AntiPOZi', 'fatpeoplehate', 'TalesofFatHate']
# final_hateful_srs = ['CoonTown', 'WhiteRights', 'Trans_fags', 'SlutJustice',
#                      'TheRedPill', 'KotakuInAction', 'IslamUnveiled', 'GasTheKikes',
#                      'AntiPOZi', 'fatpeoplehate', 'TalesofFatHate']
all_srs = final_hateful_srs + final_nothate_srs

# build sql queries to extract comments
query = []
for i in range(len(all_srs)):
    query.append("SELECT subreddit,id, name, body FROM MAY2015 WHERE subreddit = '" + all_srs[i] + "';")

# load a df with the first set of results
print "Building df"
df = pd.read_sql_query(query[0], conn)

# iterate through queries and append to dataframe
for i in range(1, len(query)):
    print "Building df"
    df = df.append(pd.read_sql_query(query[i], conn), ignore_index=True)

# Reset df index, to make it workable
df.reset_index(drop=True)

# Create a not hate label for all entries & write over with hateful labels where appropriate
df['label'] = 'NotHate'

# Need to label our comments depending on subreddit.
df.ix[(df.subreddit == 'CoonTown'), 'label'] = 'RaceHate'
df.ix[(df.subreddit == 'WhiteRights'), 'label'] = 'RaceHate'
print "Done with RaceHate"
df.ix[(df.subreddit == 'Trans_fags'), 'label'] = 'GenderHate'
df.ix[(df.subreddit == 'SlutJustice'), 'label'] = 'GenderHate'
df.ix[(df.subreddit == 'TheRedPill'), 'label'] = 'GenderHate'
# df.ix[(df.subreddit == 'KotakuInAction'), 'label'] = 'GenderHate'  # I took this one out
print "Done with GenderHate"
df.ix[(df.subreddit == 'IslamUnveiled'), 'label'] = 'ReligionHate'
df.ix[(df.subreddit == 'GasTheKikes'), 'label'] = 'ReligionHate'
df.ix[(df.subreddit == 'AntiPOZi'), 'label'] = 'ReligionHate'
print "Done with ReligionHate"
df.ix[(df.subreddit == 'fatpeoplehate'), 'label'] = 'SizeHate'
df.ix[(df.subreddit == 'TalesofFatHate'), 'label'] = 'SizeHate'

print "Done with Hate Categorization"

# save out a subset of these data to work with, so it doesn't take quite so long to run!
# 1.5 million * 25% = 375,000 --> plenty
dfsave = df.sample(frac=0.25, replace=False, weights=None, random_state=None, axis=0)

# Save file for later access
pickle.dump(dfsave, open('labeledhate_pyladies.p', 'wb'))

# # To load file:
# df = pickle.load(open('labeledhate_pyladies.p', 'rb'))

# Don't forget to close the connection!!!
conn.close()

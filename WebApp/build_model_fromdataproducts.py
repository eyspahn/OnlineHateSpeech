import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


def build_model(filename):
    ### write your code to build a model
    #read in code & build tfidf vectorizer on "body"
    documents = pd.read_csv(filename)
    tfidf_v = TfidfVectorizer(stop_words='english')
    tfidf_v_fit = tfidf_v.fit(documents.body)

    tfidf_vectorized = tfidf_v.fit_transform(documents.body)


    #build model to predict on 'section_name'
    X = tfidf_vectorized
    y = documents.section_name

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=1)

    clf = MultinomialNB()
    #clf.fit(X_train, y_train)

    vectorizer, model = tfidf_v_fit, clf.fit(X_train, y_train)
    return vectorizer, model


if __name__ == '__main__':
    vectorizer, model = build_model('data/articles.csv')
    with open('data/vectorizer.pkl', 'w') as f:
        pickle.dump(vectorizer, f)
    with open('data/model.pkl', 'w') as f:
        pickle.dump(model, f)

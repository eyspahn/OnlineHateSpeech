from flask import Flask, request, render_template
import cPickle as pickle
from numpy import max
import xgboost as xgb
from runfinalmodelpreds_v1 import predict_comment, tokenize, stem_tokens


application = Flask(__name__)


# home page
@application.route('/')
def index():

    return render_template("index.html")


# submit page, to accept text
@application.route('/submit')
def submit():

    return render_template('submit.html')


# predict page, to display prediction result
@application.route('/predict', methods = ['POST'])
def predict():
    global text
    text = str(request.form['user_input'].encode(encoding='ascii', errors='ignore'))
    pred_class, pred_prob, text = predict_comment(text, classes, bst, vect)

    return render_template('predict.html', pred_class=pred_class, pred_prob=pred_prob,
                           text=text)


if __name__ == '__main__':

    classes = ['Not Hate', 'Size Hate', 'Gender Hate', 'Race Hate', 'Religion Hate']

    # load saved xgboost model
    bst = xgb.Booster()
    bst.load_model('./hatespeech.model')
    # load vectorizer
    vect = pickle.load(open('./vect.p', 'rb'))
    application.run(debug=True)    
    # application.run(host='0.0.0.0', port=8080, debug=True)

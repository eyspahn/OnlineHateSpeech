from flask import Flask, request
import cPickle as pickle
from numpy import max

app = Flask(__name__)


# home page
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>Welcome</title>
        </head>
      <body>
        <h1>Hello, and welcome.</h1>
        <br>
        <h2>
        <a href='/submit'>Access the hate speech predictor here.</a>
        </h2>

        <p> The predictor is built on a selection of hateful and not-hateful subreddits.<p>

        </form>
      </body>
    </html>'''


# technical details
@app.route('/techdetails')
def techdetails():
    return '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <title>Details of Analysis and Model</title>
            </head>
          <body>
            <p>Placeholder: content</p>
            <p></p>
            <br> <br><br><br><br><br><br><br><br>

          </body>
        </html>'''


# submit page, to accept text
@app.route('/submit')
def submit():
    return '''
        <form action="/predict" method='POST' >
            <textarea name="user_input", rows=10, cols=80 ></textarea>
            <input type="submit" />
        </form>
        '''

#     <input type="text" name="user_input" />

# predict page, to display prediction result
@app.route('/predict', methods = ['POST'])
def predict():
    text = str(request.form['user_input'].encode(encoding='ascii', errors='ignore'))
    # X = vectorizer.transform([text])
    # pred = model.predict(X)
    # predict_proba = model.predict_proba(X)


    return '''<!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>Your Category</title>
        </head>
      <body>
        <p> Your text is predicted to be: {0}</p>
        <p>The probability this is the correct category: {1}% </p>
        <br> <br><br><br><br><br><br><br><br>

        <h4>The text you submitted: </h4>
        {2}
      </body>
    </html>'''.format(str(pred[0]), str(int(round(max(predict_proba) * 100, 0))), text)


if __name__ == '__main__':
    # with open('data/vectorizer.pkl') as f:
    #     vectorizer = pickle.load(f)
    # with open('data/model.pkl') as f:
    #     model = pickle.load(f)

    app.run(host='0.0.0.0', port=8080, debug=True)

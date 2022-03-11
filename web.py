from distutils import text_file
from distutils.command.clean import clean
from json import load
from pyexpat import model
from statistics import mode
from unittest import TestCase
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, jsonify, make_response, json
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import gensim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import model_from_json 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

app = Flask(__name__)
ps = PorterStemmer()

countVect=pickle.load(open('counteVect.pkl','rb'))
lr = pickle.load(open('logistic_regression.pkl','rb'))
tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))
svm = pickle.load(open('svm.pkl', 'rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

def cleaning(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

def LR_prediction(text):
    vect = countVect.transform([text])
    model = lr.predict(vect)

    if model == 1:
        prediction = 'REAL NEWS'
    else:
        prediction = 'FAKE NEWS'

    return prediction

def svm_prediction(text):
    vect = tfidfvect.transform([text]).toarray()
    model =  svm.predict(vect)

    if model == 1:
        prediction = 'REAL NEWS'
    else:
        prediction = 'FAKE NEWS'

    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    
    cleaned_text = cleaning(text)

    if "svm_button" in request.form:
        prediction = svm_prediction(cleaned_text)
        pass
    elif "lr_button" in request.form:
        prediction = LR_prediction(cleaned_text)
        pass

    return render_template('index.html', predict_content=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
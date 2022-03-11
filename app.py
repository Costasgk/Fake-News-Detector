import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# from wordcloud import WordCloud, STOPWORDS
import string
from sklearn import svm
from sklearn.preprocessing import LabelEncoder as le
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns
import tensorflow_hub as hub
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import classification_report

fake = pd.read_csv("Fake.csv")
true=  pd.read_csv("True.csv")

emptyt=[index for index,text in enumerate(true.text.values) if str(text).strip() == '']
true=true[true["text"].str.strip() !=''] 
emptyf=[index for index,text in enumerate(fake.text.values) if str(text).strip() == '']
true['From'] = true['text'].str.split('-' , 1).str[0]
true['text'] = true['text'].str.split('-' , 1).str[1]
true=true[true["text"].str.strip() !=''] 

true['num'] = 1
fake['num'] = 0

data = pd.concat([fake, true])


# ---------------- CLEANING ----------------

print("---------------- CLEANING ----------------")

patternDel = "http"
filter1 = data['date'].str.contains(patternDel)
data = data[~filter1]

pattern = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
filter2 = data['date'].str.contains(pattern)
data = data[filter2]

data['date'] = pd.to_datetime(data['date'])

data = data.fillna("")

datasub=data.groupby(['subject', 'num'])['text'].count()
datasub = datasub.unstack().fillna(0)

data_ = data.copy()
data_ = data_.sort_values(by = ['date'])
data_ = data_.reset_index(drop=True)

dataf = data_[data_['num'] == 0]
dataf = dataf.groupby(['date'])['num'].count()
f = pd.DataFrame(dataf)

datat = data_[data_['num'] == 1]
datat = datat.groupby(['date'])['num'].count()
t = pd.DataFrame(datat)

stopwords = nltk.corpus.stopwords.words("english")

def precleaning(x):
    f = x
    f = f.lower()
    f = re.sub('\[.*?\]', '', f) 
    f = re.sub(r'[^\w\s]','',f) 
    f = re.sub('\w*\d\w*', '', f) 
    f = re.sub(r'http\S+', '', f)
    f = re.sub('\n', '', f)
    return f

def remove_stopwords(text):
    token_text = nltk.word_tokenize(text)
    removed = [word for word in token_text if word not in stopwords]
    joinedtext = ' '.join(removed)
    return joinedtext

def cleaning(text):
    text = precleaning(text)
    text = remove_stopwords(text)
    return text

new_data = data_.copy()
new_data['text'] = data_.text.apply(lambda x : cleaning(x))
del new_data['From']

a = pd.DataFrame(new_data.text)

def common_tokens_title(data, feature, name):
    column = data[feature].str.lower() 
    text = ' '.join(column)
    exclude = set(string.punctuation)
    words = ''.join(char for char in text if char not in exclude)
    words_splitted = words.split()
    words_stopped = [word for word in words_splitted if not word in stopwords]
    # print(f'{name}:\n{pd.DataFrame(nltk.FreqDist(words_stopped).most_common(10))[0]}')

common_tokens_title(true, 'title', 'Most common descriptive words in Real News Titles')
print('\n')
common_tokens_title(fake, 'title', 'Most common descriptive words in Fake News Titles')
print('\n')
common_tokens_title(new_data, 'title', 'Most common descriptive words in Combined News Titles')

print("Cleaning is done")

# ---------------- WORD2VEC ----------------

print("---------------- WORD2VEC ----------------")

y = new_data['num'].values
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in new_data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

EMBEDDING_DIM = 100

w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)

w2v_model.save("trained_model.bin")

vocab_len = len(w2v_model.wv.vocab)

print("Vectorization is done")

# ---------------- LOGISTIC REGRESSION ----------------

print("---------------- LOGISTIC REGRESSION ----------------")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


X_train, X_test, y_train, y_test = train_test_split(new_data['text'], 
                                                    new_data['num'], 
                                                    test_size = 0.4,
                                                    random_state=0)

vect = CountVectorizer().fit(X_train)

X_train_vectorized = vect.transform(X_train)

X_train_vectorized

logistic_model = LogisticRegression()
logistic_model.fit(X_train_vectorized, y_train)

predictions = logistic_model.predict(vect.transform(X_test))

acc = metrics.accuracy_score(y_test, predictions)
f1 = metrics.f1_score(y_test, predictions,pos_label=1)

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = logistic_model.coef_[0].argsort()


print("Logistic Regression is done")

# ---------------- NEURAL NETWORKS ----------------

# # print("---------------- NEURAL NETWORKS ----------------")

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# X = tokenizer.texts_to_sequences(X)
# X = pad_sequences(X, maxlen=1000)

# word_index = tokenizer.word_index 
# vocab_size = len(word_index) + 1
# #Get the weight matrix for embedding layer
# def get_weight(model,word_index):
#     weight_matrix = np.zeros((vocab_size,EMBEDDING_DIM ))
#     for word, index in word_index.items():
#         weight_matrix[index]=model[word]
#     return weight_matrix

# emb_vec = get_weight(w2v_model.wv,word_index)

# print(emb_vec)  #1


# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# nn_model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights = [emb_vec], input_length=1000,trainable=False),
#     tf.keras.layers.LSTM(128, return_sequences=True),
#     tf.keras.layers.Dropout(0.25),
#     tf.keras.layers.LSTM(64),
#     tf.keras.layers.Dense(32,activation="relu"),
#     tf.keras.layers.Dense(1,activation="sigmoid")
# ])

# nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# nn_model.summary


# history = nn_model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test),batch_size=128)
# classification_result = nn_model.evaluate(x_test,y_test)
# print("test loss, test acc:",classification_result)  #2


# print("Neural Networks is done")


# nn_json = nn_model.to_json()

# with open("nn.json", "w") as json_file:
#     json_file.write(nn_json)

# nn_model.save_weights("nn_model.h5")
# ---------------- SVM ----------------

print("---------------- SVM ----------------")
X=new_data['text'].to_list()
y=new_data['num'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)
Vectorizer = TfidfVectorizer(max_df=0.9,ngram_range=(1, 2))
TfIdf=Vectorizer.fit(X_train)
X_train=TfIdf.transform(X_train)

svm_model =svm.LinearSVC(C=0.1)
svm_model.fit(X_train,y_train)

X_test=TfIdf.transform(X_test)
y_pred=svm_model.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, y_pred)

print("SVM is done")


# Save:
# Vectorization from Word2Vec
# Models from LR,NN,SVM

import pickle

pickle.dump(vect, open('counteVect.pkl', 'wb'))
pickle.dump(TfIdf, open('tfidfvect2.pkl', 'wb'))
pickle.dump(w2v_model, open('w2v_model.pkl', 'wb'))
pickle.dump(logistic_model, open('logistic_regression.pkl', 'wb'))
pickle.dump(svm_model, open('svm.pkl', 'wb'))

# pickle.dump(nn_model, open('nn.pkl', 'wb'))

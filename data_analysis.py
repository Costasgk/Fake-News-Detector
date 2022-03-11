import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
from sklearn import svm
from sklearn.preprocessing import LabelEncoder as le
import nltk
nltk.download('stopwords')
nltk.download('punkt')
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

fake.head()

true.head()

#Counting by Subjects 
for key,count in fake.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")
    
#total rows
print(f"Total Records:\t{fake.shape[0]}")

#Counting by Subjects 
for key,count in true.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")
    
# Total Rows
print(f"Total Records:\t{fake.shape[0]}")

emptyt=[index for index,text in enumerate(true.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(emptyt)}")
print(f"No of  total rows: {len(true.text.values)}")

true=true[true["text"].str.strip() !=''] 
print(f"No of  total rows: {len(true.text.values)}")
true

emptyf=[index for index,text in enumerate(fake.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(emptyf)}")
print(f"No of total rows: {len(fake.text.values)}")

true['From'] = true['text'].str.split('-' , 1).str[0]
true['text'] = true['text'].str.split('-' , 1).str[1]
true

true=true[true["text"].str.strip() !=''] 
print(f"No of  total rows: {len(true.text.values)}")
true

true['num'] = 1
fake['num'] = 0


data = pd.concat([fake, true])


plt.figure(figsize = (12,8))
sns.set_style("darkgrid")
sns.countplot(data.num)
plt.show()
# by_tf = data.num.value_counts()
# by_tf.plot(kind='bar')

plt.figure(figsize = (12,8))
plt.pie(data["num"].value_counts().values,explode=[0,0],labels=data.num.value_counts().index, autopct='%1.1f%%')

# data.isna().sum()
# data.title.count()
data.subject.value_counts()

plt.figure(figsize = (12,8))
ax = sns.countplot(x="subject",  hue='num', data=data)

"""# Cleaning"""

patternDel = "http"
filter1 = data['date'].str.contains(patternDel)
data = data[~filter1]
# Διαγράφω ο,τι είναι link.

pattern = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
filter2 = data['date'].str.contains(pattern)
data = data[filter2]

data['date'] = pd.to_datetime(data['date'])

data = data.fillna("")
data.head(15000)

datasub=data.groupby(['subject', 'num'])['text'].count()
datasub = datasub.unstack().fillna(0)
datasub

ax = datasub.plot(kind = 'bar', figsize = (12,8), grid = True)
plt.show()

data_ = data.copy()
data_ = data_.sort_values(by = ['date'])
data_ = data_.reset_index(drop=True)
data_

# Σορτάρισμα με βάση ημερομηνία

# Μετράω τα fake news ανα ημέρα
dataf = data_[data_['num'] == 0]
dataf = dataf.groupby(['date'])['num'].count()
f = pd.DataFrame(dataf)
f

# Μετράω τα real news ανα ημέρα
datat = data_[data_['num'] == 1]
datat = datat.groupby(['date'])['num'].count()
t = pd.DataFrame(datat)
# t

# import plotly.offline as pyoff
# import plotly.graph_objs as go
# import plotly.graph_objs as go

# plot_data = [
#     go.Scatter(
#         x=t.index,
#         y=t['num'],
#         name='True',
#         #x_axis="OTI",
#         #y_axis="time",
#     ),
#     go.Scatter(
#         x=f.index,
#         y=f['num'],
#         name='Fake'
#     )
    
# ]
# plot_layout = go.Layout(
#         title='Day-wise',
#         yaxis_title='Number',
#         xaxis_title='Time',
#         plot_bgcolor='rgba(0,0,0,0)'
#     )
# fig = go.Figure(data=plot_data, layout=plot_layout)
# pyoff.plot(fig)

# plt.figure(figsize = (20,20)) # Text that is not Real
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.num == 1].text))
# plt.axis("off")
# plt.imshow(wc , interpolation = 'bilinear')
# plt.show()

# plt.figure(figsize = (20,20)) # Text that is not Real
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.num == 0].text))
# plt.axis("off")
# plt.imshow(wc , interpolation = 'bilinear')
# plt.show()

stopwords = nltk.corpus.stopwords.words("english")

# Καθαρίζει τα κείμενα στο 'text' από τα σημεία στίξης,παρενθέσεις κτλ
def precleaning(x):
    f = x
    f = f.lower()
    f = re.sub('\[.*?\]', '', f) 
    f = re.sub(r'[^\w\s]','',f) 
    f = re.sub('\w*\d\w*', '', f) 
    f = re.sub(r'http\S+', '', f)
    f = re.sub('\n', '', f)
    return f

# Αφαιρεί τις stopwords
def remove_stopwords(text):
    token_text = nltk.word_tokenize(text)
    removed = [word for word in token_text if word not in stopwords]
    joinedtext = ' '.join(removed)
    return joinedtext

# Συνολικό καθάρισμα
def cleaning(text):
    text = precleaning(text)
    text = remove_stopwords(text)
    return text

new_data = data_.copy()
new_data['text'] = data_.text.apply(lambda x : cleaning(x))
del new_data['From']
new_data.head()

new_data.head(12500)

a = pd.DataFrame(new_data.text)
a.head()

# stopwords = set(STOPWORDS)

def common_tokens_title(data, feature, name):
    column = data[feature].str.lower() 
    text = ' '.join(column)
    exclude = set(string.punctuation)
    words = ''.join(char for char in text if char not in exclude)
    words_splitted = words.split()
    words_stopped = [word for word in words_splitted if not word in stopwords]
    print(f'{name}:\n{pd.DataFrame(nltk.FreqDist(words_stopped).most_common(10))[0]}')
    
common_tokens_title(true, 'title', 'Most common descriptive words in Real News Titles')
print('\n')
common_tokens_title(fake, 'title', 'Most common descriptive words in Fake News Titles')
print('\n')
common_tokens_title(new_data, 'title', 'Most common descriptive words in Combined News Titles')

# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(new_data[new_data.num == 1].text))
# plt.axis("off")
# plt.imshow(wc , interpolation = 'bilinear')
# plt.show()

# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(new_data[new_data.num == 0].text))
# plt.axis("off")
# plt.imshow(wc , interpolation = 'bilinear')
# plt.show()

def get_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    words = vec.transform(corpus)
    total_words = words.sum(axis=0) 
    words_freq = [(word, total_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def graph_ngrams(data_set , ngrams):
    plt.figure(figsize = (16,9))
    gram = get_ngrams(data_set.text,10,ngrams)
    gram = dict(gram)
    for key in gram:
      print(key, ' : ', gram[key])
    sns.barplot(x=list(gram.values()),y=list(gram.keys()))

"""**Μπορουμε να δημιουργήσουμε μια συναρτηση(graph_ngrams) για να πλοτάρει τα δεδομένα, αντι να κάνουμε copy-paste.**"""

# Εμφάνιζει unigrams
# graph_ngrams(new_data, 1)


# # plt.figure(figsize = (16,9))
# # uni = get_ngrams(new_data.text,10,1)  #Εμφανίζει τα δέκα πρώτα πιο συχνά unigrams
# # uni = dict(uni)
# # sns.barplot(x=list(uni.values()),y=list(uni.keys()))

# # Εμφανίζει bigrams
# graph_ngrams(new_data, 2)

# # plt.figure(figsize = (16,9))
# # bi = get_ngrams(new_data.text,10,2)  #Εμφανίζει τα δέκα πρώτα πιο συχνά bigrams
# # bi = dict(bi)
# # sns.barplot(x=list(bi.values()),y=list(bi.keys()))

# graph_ngrams(new_data, 3)

# plt.figure(figsize = (16,9))
# tri = get_ngrams(new_data.text,10,3) #Εμφανίζει τα δέκα πρώτα πιο συχνά trigrams
# tri = dict(tri)
# sns.barplot(x=list(tri.values()),y=list(tri.keys()))

"""**Μπορούμε να βάλουμε και unigram, bigram, trigram και για τα True News(datat) και Fake News(dataf) ξεχωριστά**"""

new_data.head()

"""# Word2Vec"""

y = new_data['num'].values
#μετατροπή του χ σε μορφή που θα μπορούμε να χρησιμοποιήσουμε
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

#διαστάσεις
EMBEDDING_DIM = 100


w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)

# vocab_len = len(w2v_model.wv.vocab)
# print("The number of words in the model's vocabulary:", vocab_len)

w2v_model["trump"]

w2v_model["president"]

# Παραδείγματα για το πως δουλεύει το w2v_model με τις λέξεις trump και president

w2v_model.wv.most_similar("trump")

w2v_model.wv.most_similar("york")

w2v_model.wv.most_similar("news")

# Εύρεση λέξεων που "μοιάζουν" με τις λέξεις trump, york, news

new_data.head()

"""# Logistic Regression"""

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

# Μετατρέπει ό,τι έιναι μορφή text σε δίανυσμα

X_train_vectorized = vect.transform(X_train)

X_train_vectorized

logistic_model = LogisticRegression()
logistic_model.fit(X_train_vectorized, y_train)
# Εκπάιδευση του μοντέλου με Logistic Regression

predictions = logistic_model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

# ROC curve = γράφος που δείχνει την απόδοση του classification
# AUC:Area Under the ROC Curve

print(metrics.confusion_matrix(y_test, predictions, labels=[0, 1]))  # true = 1, fake = 0 . Εκτυπώνει [[TN,FP][FN,TP]]
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_test, predictions, labels=[0, 1]))

from mlxtend.plotting import plot_confusion_matrix

acc = metrics.accuracy_score(y_test, predictions)
f1 = metrics.f1_score(y_test, predictions,pos_label=1)

print('The accuracy of the prediction is {:.2f}%.\n'.format(acc*100))
print('The F1 score is {:.3f}.\n'.format(f1))

cmatrix = confusion_matrix(y_test, predictions , labels=[0, 1])
plot_confusion_matrix(cmatrix,
                      show_normed=True, colorbar=True,
                      )


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions , labels=[0, 1])
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax= ax, fmt='d', cmap="YlGn")
ax.set_ylabel('y_test')
ax.set_xlabel('predictions')

"""Να κρατήσουμε ένα απο τα δύο confusion matrixes"""

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model
sorted_coef_index = logistic_model.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefficients:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefficients: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# """# Neural Networks"""

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# X = tokenizer.texts_to_sequences(X)
# X = pad_sequences(X, maxlen=1000)


# # Μετατρέπει τα text σε tokens.

# word_index = tokenizer.word_index 
# vocab_size = len(word_index) + 1
# #Get the weight matrix for embedding layer
# def get_weight(model,word_index):
#     weight_matrix = np.zeros((vocab_size,EMBEDDING_DIM ))
#     for word, index in word_index.items():
#         weight_matrix[index]=model[word]
#     return weight_matrix

# emb_vec = get_weight(w2v_model.wv,word_index)

# print(emb_vec)

# from sklearn.model_selection import train_test_split
# import tensorflow as tf

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

# history = nn_model.fit(x_train,y_train,epochs=2,validation_data=(x_test,y_test),batch_size=128)
# classification_result = nn_model.evaluate(x_test,y_test)
# print("test loss, test acc:",classification_result)

"""# SVM"""

X=new_data['text'].to_list()
y=new_data['num'].to_list()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=10)
Vectorizer = TfidfVectorizer(max_df=0.9,ngram_range=(1, 2))
TfIdf=Vectorizer.fit(X_train)
X_train=TfIdf.transform(X_train)

# training  
svm_model =svm.LinearSVC(C=0.1)
svm_model.fit(X_train,y_train)

# evaluation
X_test=TfIdf.transform(X_test)
y_pred=svm_model.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, y_pred)
print('The accuracy of the prediction is {:.2f}%.\n'.format(svm_acc*100))

print(classification_report(y_test, y_pred))


# Save:
# Vectorization from Word2Vec
# Models from LR,NN,SVM

import pickle

pickle.dump(w2v_model, open('w2v_model.pkl', 'wb'))
pickle.dump(logistic_model, open('logistic_regression.pkl', 'wb'))
pickle.dump(svm_model, open('svm.pkl', 'wb'))

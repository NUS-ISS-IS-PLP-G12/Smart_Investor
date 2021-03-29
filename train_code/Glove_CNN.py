import pandas as pd
import numpy as np

from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D,Reshape, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.models import load_model

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

import matplotlib.pyplot as plt
from plot_keras_history import plot_history

import pandas as pd
# news=pd.read_table("text3.csv",names = ["text", "label"])
# #,header=None,names = ["Class", "Text"]
# news.head()
# # a = news.groupby("Class")
# # a.head()
# # a.describe()
# X = news.loc[:,'text']
# y = news.loc[:,'label']
# print(news)
# X_train = news.iloc[0:1391,0]
# X_test  = news.iloc[1391:1988,0]
# y_train= news.iloc[0:1391,1]
# y_test= news.iloc[1391:1988,1]
# print(X_train)
import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
from sklearn.model_selection import cross_val_score
import string
from sklearn.model_selection import RandomizedSearchCV
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import codecs

nltk.download('stopwords')


stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")


stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+|url\S+|URL\S+')
    return url.sub(r'',str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',str(text))


def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

# general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text
news=pd.read_table("text3.csv",names = ["text", "label"],encoding= 'unicode_escape')
#,header=None,names = ["Class", "Text"]
news.head()
# a = news.groupby("Class")
# a.head()
# a.describe()
X = news.loc[:,'text']
y = news.loc[:,'label']
print(news)
print(X)
print(y)

X=X.map(remove_URL)
X=X.map(remove_emoji)
X=X.map(remove_html)
X=X.map(remove_punctuation)
X=X.map(decontracted)
X=X.map(final_preprocess)
print(X)

# #tokenization
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X)
# X=tokenizer.texts_to_sequences(X)

# word_index = tokenizer.word_index

import numpy as np
print(type(X))
X=np.array(list(X)).tostring()
print(type(X))
X = X.decode(encoding='utf-8', errors="ignore")
print(type(X))

embeddings_index = np.load('glove_dict.npy', allow_pickle=True).item()
    # f = open('glove.840B.300d.txt')
    # for line in f:
    #     values = line.split(' ')
    #     word = values[0] ## The first entry is the word
    #     coefs = np.asarray(values[1:], dtype='float32') ## These are the vectors representing the embedding for the word
    #     embeddings_index[word] = coefs
    # f.close()
glove_words =  set(embeddings_index.keys())

converted_data = []

for i in range(0, news.shape[0]):
        vector = np.zeros(300) # as word vectors are of zero length
        cnt_words =0; # num of words with a valid vector in the sentence
        for word in X[i].split():
            if word in glove_words:
                vector += embeddings_index[word]
                cnt_words += 1
        if cnt_words != 0:
            vector /= cnt_words
        converted_data.append(vector)


X = pd.DataFrame(converted_data)
print(X)


model = Sequential()
model.add(Embedding(input_dim=36214, output_dim=100,trainable=False))
model.add(Conv1D(512, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary()

model.fit(X,y,batch_size=30,epochs=10)
# import pickle 
import joblib
from sklearn.svm import SVC
from sklearn import datasets
import pickle as p
with open('CNN.pickle','wb') as fw:
    p.dump(model,fw)

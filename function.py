import numpy as np
import sys
import pandas as pd
import os
import re 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import string
from keras.preprocessing.text import Tokenizer, one_hot
import sklearn


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

def cleaning(data):

    data['text']=data['text'].map(remove_URL)
    data['text']=data['text'].map(remove_emoji)
    data['text']=data['text'].map(remove_html)
    data['text']=data['text'].map(remove_punctuation)
    data['text']=data['text'].map(decontracted)
    data['text']=data['text'].map(final_preprocess)

    import os
    sss = os.path.join(os.path.dirname(__file__))

    embeddings_index = np.load(sss+'/glove_dict.npy', allow_pickle=True).item()

    glove_words =  set(embeddings_index.keys())

    '''
    Below is a uliity function that takes sentenes as a input and return the vector representation of the same
    Method adopted is similar to average word2vec. Where i am summing up all the vector representation of the words from the glove and 
    then taking the average by dividing with the number of words involved
    '''

    converted_data = []

    for i in range(0, data.shape[0]):
            vector = np.zeros(300) # as word vectors are of zero length
            cnt_words =0; # num of words with a valid vector in the sentence
            for word in data['text'][i].split():
                if word in glove_words:
                    vector += embeddings_index[word]
                    cnt_words += 1
            if cnt_words != 0:
                vector /= cnt_words
            converted_data.append(vector)


    a = pd.DataFrame(converted_data)
    return a

import pickle

def mlp(data0):
    # Load from file
    import os
    # sss = os.path.join(os.path.dirname(__file__))
    pkl_filename = os.path.join(os.path.dirname(__file__))+"/pickle/MLP.pickle"
    with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
            
    data0 = data0.iloc[:, :746].values
    y_predict = pickle_model.predict_proba(data0)
    #np.savetxt("proba_lr.txt",y_predict)
    possi_mlp = y_predict
    possi_mlp=possi_mlp.tolist()
    possi_mlp = possi_mlp[0]
    return possi_mlp

def cnn(data1):
    import os
    pkl_filename = os.path.join(os.path.dirname(__file__)) + "/pickle/CNN.pickle"
    with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
    y_train_predict = pickle_model.predict(data1)
    # print(y_train_predict)
    possi_cnn = y_train_predict
    possi_cnn = possi_cnn.tolist()
    possi_cnn = possi_cnn[0]
    a=1-possi_cnn[0]#=0的概率
    #print(a)
    possi_cnn.insert(0, a)
    return possi_cnn

def lstm(data2):
    import os
    pkl_filename = os.path.join(os.path.dirname(__file__)) + "/pickle/LSTM.pickle"
    with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
    y_train_predict = pickle_model.predict(data2)
    # print(y_train_predict)
    possi_ls = y_train_predict
    possi_ls = possi_ls.tolist()
    possi_ls = possi_ls[0]
    b=1-possi_ls[0]#=0的概率
    #print(b)
    possi_ls.insert(0, b)
    #print(possi_ls)
    return possi_ls

def rf(data3):
    X = data3.iloc[:,0:4]
    y = data3.loc[:,'label']
    # Load from file
    pkl_filename = os.path.join(os.path.dirname(__file__)) + "/pickle/num_RF.pickle"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    y_predict = pickle_model.predict_proba(X)
    # np.savetxt("proba_RF.txt",y_predict)
    possi_rf = y_predict
    possi_rf=possi_rf.tolist()
    possi_rf = possi_rf[0]
    return possi_rf

def lr(data4):
# Load from file
    X = data4.iloc[:,0:4]
    y = data4.loc[:,'label']
    pkl_filename = os.path.join(os.path.dirname(__file__)) + "/pickle/num_lr.pickle"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    y_predict = pickle_model.predict_proba(X)
    possi_lr = y_predict
    possi_lr=possi_lr.tolist()
    possi_lr = possi_lr[0]
    return possi_lr


import numpy as np
import sys
import pandas as pd
import os
import re 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk

from function import mlp
from function import cnn
from function import lstm
from function import cleaning
from function import lr
from function import rf

nltk.download('stopwords')


stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")

# pip install transformers
# pip install tensorflow==2.1.0
# pip install simpletransformers
# pip install tokenizers==0.8.1.rc1
# export CUDA_HOME=/usr/local/cuda-10.1
# git clone https://github.com/NVIDIA/apex
# %cd apex
# pip install -v --no-cache-dir ./

import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# -*- coding: utf-8 -*-
import pickle

from crawl_amzn_news_stocks_worldnews import get_stock_ohlv,get_txt

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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    e_y = e_x / e_x.sum()
    return e_y

def be(df):
    #===========================bert======================================#
    import spacy, re
    import numpy as np
    import pandas as pd 

    # df = np.empty([0,0], dtype = str)
    # df['text'] = txt_news

    #Data Cleanup
    df['text']=df['text'].str.replace('\n','')
    df['text']=df['text'].str.replace('\r','')
    df['text']=df['text'].str.replace('\t','')
    
    #This removes unwanted texts
    df['text'] = df['text'].apply(lambda x: re.sub(r'[0-9]','',str(x)))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;.:-]',' ',str(x)))
    
    #Converting all upper case to lower case
    df['text']= df['text'].apply(lambda s:s.lower() if type(s) == str else s)
    

    #Remove un necessary white space
    df['text']=df['text'].str.replace('  ',' ')

    #Remove Stop words
    nlp=spacy.load("en_core_web_sm")
    df['text'] =df['text'].apply(lambda x: ' '.join([word for word in x.split() if nlp.vocab[word].is_stop==False ]))

    #===========================bert======model================================#
    # """Compute softmax values for each sets of scores in x.""" #


    from simpletransformers.classification import ClassificationModel
    import os


    sss = os.path.join(os.path.dirname(__file__))
    # print(sss)

    model = ClassificationModel('bert', sss+'/bert', num_labels=2, args={'fp16': False,'overwrite_output_dir': True,'output_dir':'bert_classifier_model',"train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,'num_train_epochs': 1}, use_cuda=False)
    # model = ClassificationModel('bert', '/Users/taoxiyan/Downloads/plp/plp_project/final_code/Smart_Investor/bert', num_labels=2, args={'fp16': False,'overwrite_output_dir': True,'output_dir':'bert_classifier_model',"train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,'num_train_epochs': 1}, use_cuda=False)
    # model = ClassificationModel('bert', './bert', num_labels=2, args={'fp16': False,'overwrite_output_dir': True,'output_dir':'bert_classifier_model',"train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,'num_train_epochs': 1}, use_cuda=False)

    df = df.values.tolist()
    df = df[0]
    result, model_outputs,atten_score = model.predict(df)      

    # vocab = open( sss + '/../bert/vocab.txt' ,'r', encoding='utf-8').readlines()
    # dic_for_bert = {}
    # for inx,word in enumerate(vocab):
    #     dic_for_bert[inx]=word.strip()
    # word_list = return_highlight_word(atten_score,dic_for_bert,df)

    import numpy as np
    preds = [np.argmax(tuple(m)) for m in model_outputs]
    possibility = [tuple(k) for k in model_outputs]

    possi_b = []
    for m in possibility:
        mm = softmax(m)
        possi_b.append(mm)
    # print(type(possi_be))
    # print(possi_be[0])

    possi_b = possi_b[0]
    possi_b=possi_b.tolist()

    return possi_b

def get_final():
    text = get_txt()
    a = {'text':[text]}
    df = pd.DataFrame(a)

    l_be,r_be = be(df)

    b = {'text':[text],'label':[0]}
    df = pd.DataFrame(b)
    news = df

    oo,hh,ll,vv = get_stock_ohlv()
    b = {'open':[oo],'high':[hh],'low':[ll],'volume':[vv],'label':[0]}
    num = pd.DataFrame(b)

    df=cleaning(news)
    possi_mlp=mlp(df)
    # possi_cnn=cnn(df)
    possi_cnn = [0,0]
    # possi_ls=lstm(df)
    possi_ls = [0,0]
    possi_rf=rf(num)
    possi_lr=lr(num)

    possi1 = np.append(possi_cnn,possi_rf)
    possi2 = np.append(possi_mlp,possi_ls)
    possi3 = np.append(possi1,possi2)
    possi  = np.append(possi3,possi_lr)
    equation_inputs = possi

    k = 0
    product = np.zeros((1,equation_inputs.shape[0]), dtype = float, order = 'C') #create an empty array

    while k<10:

        prod = equation_inputs[k]
        product[0][k] = prod
        k=k+1


    product = product.T

    l_p = product[0]+product[2]+product[4]+product[6]+product[8]+l_be
    r_p = product[1]+product[3]+product[5]+product[7]+product[9]+r_be

    # l_p = product[0]+product[2]+product[4]+product[6]+l_be
    # r_p = product[1]+product[3]+product[5]+product[7]+r_be

    l_pp = l_p / (l_p+r_p)
    r_pp = r_p / (l_p+r_p)
    presult = np.append(l_p,r_p)
    presult_p = np.append(l_pp,r_pp)

    prediction_last = np.argmax(presult)

    return prediction_last,presult_p

# a,b = get_final()
# print("final result =",a,"\n","possibility = ",b)

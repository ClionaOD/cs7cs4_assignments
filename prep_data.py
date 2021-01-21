# -*- coding: utf-8 -*-
"""
@author: Cliona O'Doherty

Take raw data from https://www.scss.tcd.ie/Doug.Leith/CSU44061/X2020/final.php
Translate and structure into a features dataframe
""" 

import os
import requests
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from googletrans import Translator
from nltk.classify import TextCat

import nltk
nltk.download('crubadan')

# load the data, construct features
X = [] ; y = [] ; z = []

with open('./reviews_44.jl','r') as f:
    for item in jsonlines.Reader(f):
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

# structure the data in dataframe format for later ease of access
df = pd.DataFrame(columns=['text','text_translate','voted_up','early_access'])
df['text'] = pd.Series(X)
df['voted_up'] = pd.Series(y)
df['early_access'] = pd.Series(z)

# translate non-english reviews
X_same = {}
X_trans = {}
for idx, text in enumerate(X):
    lang = TextCat().guess_language(text)
    if lang == 'eng':
        X_same[idx] = text
    else:
        X_trans[idx] = text

trans_df = pd.DataFrame.from_dict([X_trans])

translator = Translator()
translations = translator.translate(list(trans_df[0,:].values))
translations = [translation.text for translation in translations]

trans_df.loc[1,:] = translations
X_trans = {k:v[1] for k,v in trans_df.to_dict().items()}

for idx, text_dict in translations.items():
    

for k,v in X_same.items():
    df.loc[k,'text_translate'] = v
for k,v in X_trans.items():
    df.loc[k,'text_translate'] = v

#save the data and translations
df.to_csv('./dataset_translated_df.csv')

#df = pd.read_csv('./dataset_translated_df.csv', index_col=0)
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
translator = Translator()
X_trans = translator.translate(X)
X_trans = [x.text for x in X_trans]

df['text_translate'] = pd.Series(X_trans)

#save the data and translations
df.to_csv('./dataset_translated_df.csv')
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')

data = pd.read_csv('./dataset_translated_df.csv')

X = data['text']

tokenizer = WhitespaceTokenizer().tokenize
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=nltk.corpus.stopwords.words('english'))
X_fit = vectorizer.fit_transform(X) 

df = pd.DataFrame(X_fit.toarray(),columns=vectorizer.get_feature_names())







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import WhitespaceTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier


data = pd.read_csv('./dataset_translated_df.csv',index_col=0)

#X = data['text_translate']
X = data['text']
y = data['voted_up']
z = data['early_access']

tokenizer = WhitespaceTokenizer().tokenize
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=nltk.corpus.stopwords.words('english'), max_df=0.2)
X_fit = vectorizer.fit_transform(X) 

df = pd.DataFrame(X_fit.toarray(),columns=vectorizer.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#i. Choose c
def choose_C_cv(X, y, c_range, plot_color):
    '''Implement 5 fold cross validation for testing 
    regression model (lasso or ridge) and plot results'''
    
    #Param setup
    kf = KFold(n_splits = 5)
    mean_f1 =[]; std_f1 =[]
       
    #Loop through each k fold
    for c_param in c_range:
        print('C = {}'.format(c_param))
        count = 0; f1_temp = [] 
        model = LogisticRegression(penalty= 'l2', C = c_param)
                
        for train_index, test_index in kf.split(X):           
            count = count + 1 
            print('count kf = {}'.format(count))
            model.fit(X.iloc[list(train_index)], y[train_index])
            ypred = model.predict(X.iloc[list(test_index)])
            f1X = f1_score(y[test_index],ypred)
            #mse = mean_squared_error(y[test_index],ypred)
            f1_temp.append(f1X)
        
        #Get mean & variance
        mean_f1.append(np.array(f1_temp).mean())
        std_f1.append(np.array(f1_temp).std())
        
    #Plot
    plt.errorbar(c_range, mean_f1, yerr=std_f1, color = plot_color)
    plt.xlabel('C')
    plt.ylabel('Mean F1 score')
    plt.title('Choice of C in Logistic regression - 5 fold CV')
    plt.show()
    
# cross validate
c_range = [0.001, 0.01, 1, 10, 30, 50, 100, 500, 1000]
plot_color = 'g' 

#choose_C_cv(X, y, c_range, plot_color)

#Run Logistic regression
def run_logistic(Xtrain, Xtest, ytrain, ytest):
    log_reg_model = LogisticRegression(penalty= 'l2')
    log_reg_model.fit(Xtrain, ytrain)

    #log_reg_model.intercept_
    #log_reg_model.coef_
    #Predictions
    predictions = log_reg_model.predict(Xtest)

    #Performance
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))
    
    #Auc
    scores = log_reg_model.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores[:, 1])
    print('AUC = {}'.format(auc(fpr, tpr)))

    return log_reg_model

# Run the logistic regression model 
# i. Use the matching gender's features
log_reg_model = run_logistic(X_train, X_test, y_train, y_test)









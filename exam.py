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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score, roc_curve, auc
from sklearn.dummy import DummyClassifier

from scipy.spatial.distance import pdist

def construct_features(data_path, prediction_class='voted_up', translated=False):
    data = pd.read_csv(data_path,index_col=0)

    if translated:
        X = data['text_translate']
    else:
        X = data['text']
    
    if not prediction_class in data.columns:
        raise ValueError('The requested class is not in the dataset.')
    
    y = data[prediction_class]

    tokenizer = WhitespaceTokenizer().tokenize
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=nltk.corpus.stopwords.words('english'), max_df=0.2)
    X_fit = vectorizer.fit_transform(X) 

    X_df = pd.DataFrame(X_fit.toarray(),columns=vectorizer.get_feature_names())

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

###################################
# Functions for logistic regression
###################################

def cross_val_C(X, y, iter_list, range_size='', prediction_class='voted_up'):
    cv_df = pd.DataFrame(index=['mean f1', 'variance'], columns=iter_list)
    
    for c in iter_list:
        print(f'C = {c}')
        model = LogisticRegression(penalty= 'l2', C=c)
        
        kf = KFold(n_splits=5)

        f1s = []
        count = 0
        for train, test in kf.split(X):
            count += 1
            print(f'fold number {count}')
            model =  model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            f1 = f1_score(y[test],ypred)
            f1s.append(f1)
        
        mean_f1 = np.mean(np.array(f1s))
        std = np.std(np.array(f1s))
        
        cv_df.loc['mean f1',c] = mean_f1
        cv_df.loc['variance',c] = std

    fig, ax = plt.subplots()
    ax.errorbar(np.array(cv_df.columns), np.array(cv_df.loc['mean f1']), yerr=cv_df.loc['variance'])
    ax.set_xlabel('value for C')
    ax.set_ylabel('mean f1 score (nfolds=5)')
    ax.set_title(f'CV results for {range_size} range')
    plt.savefig(f'./cv_logistic_{range_size}_{prediction_class}.png')
    #plt.show()
    plt.close()
    
def run_logistic(Xtrain, Xtest, ytrain, ytest, c_param=1.0):
    log_reg_model = LogisticRegression(penalty= 'l2', C=c_param)
    log_reg_model.fit(Xtrain, ytrain)

    #Predictions
    predictions = log_reg_model.predict(Xtest)

    #Performance
    print('Logistic regression results:')
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))

    return log_reg_model

####################################
# Functions for k nearest neighbours
####################################

def CV_kNN(X, y, iter_list, prediction_class='voted_up'):
    cv_df = pd.DataFrame(index=['mean f1', 'variance'], columns=iter_list)
    
    for i in iter_list:
        print(f'K = {i}')
        
        model = KNeighborsClassifier(n_neighbors=int(i) , weights='distance')
        
        kf = KFold(n_splits=5)
        f1s = []
        count = 0
        for train, test in kf.split(X):
            count += 1
            print(f'fold number {count}')
            
            model =  model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            f1 = f1_score(y[test],ypred)
            f1s.append(f1)
        
        mean_f1 = np.mean(np.array(f1s))
        std = np.std(np.array(f1s))
        
        cv_df.loc['mean f1',i] = mean_f1
        cv_df.loc['variance',i] = std

    cv_df.to_csv('./kNN results.csv')

    fig, ax = plt.subplots()
    ax.errorbar(np.array(cv_df.columns), np.array(cv_df.loc['mean f1']), yerr=cv_df.loc['variance'])
    ax.set_xlabel('number of neighbours (K)')
    ax.set_ylabel('mean f1 score (nfolds=5)')
    ax.set_title(f'CV results for kNN')
    plt.savefig(f'./cv_kNN_{prediction_class}.png')
    #plt.show()
    plt.close()

def run_knn(Xtrain, Xtest, ytrain, ytest, k=5):
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')
    print('fitting kNN ...')
    knn_model.fit(Xtrain, ytrain)
    
    #Predictions
    print('predicting from kNN model...')
    predictions = knn_model.predict(Xtest)

    #Performance
    print('kNN results:')
    print(confusion_matrix(ytest, predictions))
    print(classification_report(ytest, predictions))

    return knn_model

#################################################
# Functions for baseline model and ROC comparison
#################################################

def run_dummy(Xtrain, Xtest, ytrain, ytest):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(Xtrain, ytrain)
    predictions_dummy = dummy_clf.predict(Xtest)

    #Evaluation
    print('Dummy results:')
    print(confusion_matrix(ytest, predictions_dummy))
    print(classification_report(ytest, predictions_dummy))

    return dummy_clf

def plot_roc_models(Xtest, ytest, log_reg_model, knn_model, dummy_clf, prediction_class='voted_up'):
    'Plot ROC Curve of implemented models'
    
    #Logistic Regression model
    scores = log_reg_model.decision_function(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores)
    plt.plot(fpr,tpr, label = 'Logistic Regression')
    print('AUC = {}'.format(auc(fpr, tpr)))

    #kNN model
    scores = knn_model.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores[:, 1])
    plt.plot(fpr,tpr, color = 'r', label = 'kNN')
    print('AUC = {}'.format(auc(fpr, tpr)))

    #Baseline Model
    scores_bl = dummy_clf.predict_proba(Xtest)
    fpr, tpr, _= roc_curve(ytest, scores_bl[:, 1])
    plt.plot(fpr,tpr, color = 'orange', label = 'Baseline Model')
    print('AUC = {}'.format(auc(fpr, tpr)))
    
    #Random Choice
    plt.plot([0, 1], [0, 1],'g--', label='Random Classifier') 

    #Labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')

    plt.legend() 
    plt.savefig(f'./roc_{prediction_class}.png')
    #plt.show()
    plt.close()

def run_modelling(data_path, prediction_class='voted_up', translated=False, cross_val=False):
    
    # i. construct features
    X_train, X_test, y_train, y_test = construct_features(data_path, prediction_class=prediction_class)

    # ii. Logistic regression
    #  Cross Validate C
    # a. choose wide range to deterimine appropriate range
    if cross_val:
        c_range = np.geomspace(0.0001,10, 10)
        cross_val_C(X_train.values, y_train.values, c_range, range_size='wide')

        # b. narrow range to find optimal value based on i
        c_range = [0.01, 0.1, 0.5, 1.0, 1.5]
        cross_val_C(X_train.values, y_train.values, c_range, range_size='narrow')

    # c. run the logistic regression model - C term = 0.01 from CV results
    log_reg_model = run_logistic(X_train, X_test, y_train, y_test, c_param=1.0)

    # iii. kNN Classifier
    # a. cross val for K
    if cross_val:
        k_range = [2,6,10,12,15]
        CV_kNN(X_train.values, y_train.values, k_range)

    # b. run the kNN model, set n_neighbours (k) = from CV results
    knn_model = run_knn(X_train, X_test, y_train, y_test, k=14)

    # iv. Dummy Classifier
    dummy_clf = run_dummy(X_train, X_test, y_train, y_test)

    #Compare performance - ROC curve
    plot_roc_models(X_test, y_test, log_reg_model, knn_model, dummy_clf)

if __name__ == "__main__":
    data_path = './dataset_translated_df.csv'
    
    run_modelling(data_path, prediction_class='voted_up', cross_val=False)

    #run_modelling(data_path, prediction_class='early_access', cross_val=True)









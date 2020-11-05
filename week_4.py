import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week4.php'
dataset = requests.get(dataset_url)
dataset = dataset.text.split('#')
with open('./week4_dataset1.txt','w') as f:
    f.write(dataset[1])
with open('./week4_dataset2.txt','w') as f:
    f.write(dataset[2])"""

def split_vis_data(data_df, save_path=False):
    X1 = data_df.iloc[:,0]
    X2 = data_df.iloc[:,1]
    X = np.column_stack((X1,X2))
    y = np.array(data_df.iloc[:,2])

    #visualise the data
    df_pos = data_df[data_df['label'] == 1]
    df_neg = data_df[data_df['label'] == -1]

    plt.scatter(df_pos.iloc[:,0],df_pos.iloc[:,1], marker='o', color='b', s=18)
    plt.scatter(df_neg.iloc[:,0],df_neg.iloc[:,1], marker='+', color='purple', s=18)
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.legend(['target = +1','target = -1'], loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

    return X, y

def CV_polynomial(X, y, iter_list, model):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    model = model
    
    kf = KFold(n_splits=5)
    
    for i in iter_list:

        Xpoly = PolynomialFeatures(degree=i).fit_transform(X)

        errors = []
        for train, test in kf.split(Xpoly):
            model =  model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
                
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',i] = mean_err
        cv_df.loc['variance',i] = std

    return cv_df

def CV_C(X, y, iter_list):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    
    for i in iter_list:
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=i)
        
        kf = KFold(n_splits=5)
        errors = []
        for train, test in kf.split(X):
            model =  model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',i] = mean_err
        cv_df.loc['variance',i] = std

    return cv_df

def plot_CV(cv_df, xlabel, title, save_path=False):
    fig, ax = plt.subplots()
    ax.errorbar(
        np.array(cv_df.columns), 
        np.array(cv_df.loc['mean error']), 
        yerr=cv_df.loc['variance']
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('mean squared error')
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_train_predict(train_df, pred_df, save_path=False):
    train_pos = train_df[train_df['label'] == 1]
    train_neg = train_df[train_df['label'] == -1]
    
    pred_pos = pred_df[pred_df['y_pred'] == 1]
    pred_neg = pred_df[pred_df['y_pred'] == -1]

    plt.scatter(train_pos.iloc[:,0], train_pos.iloc[:,1], marker='o', color='b', s=18)
    plt.scatter(train_neg.iloc[:,0], train_neg.iloc[:,1], marker='+', color='purple', s=18)
    plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
    plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)

    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.legend(['target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def get_pred_df(X, preds):
        pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
        pred_df['X1'] = X[:,0]
        pred_df['X2'] = X[:,1]
        pred_df['y_pred'] = preds
        return pred_df

def CV_kNN(X, y, iter_list):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    
    for i in iter_list:
        model = KNeighborsClassifier(n_neighbors=int(i) , weights='distance')
        
        kf = KFold(n_splits=5)
        errors = []
        for train, test in kf.split(X):
            model =  model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',i] = mean_err
        cv_df.loc['variance',i] = std

    return cv_df

def evaluate(X_train, y_train, X_test, y_test, y_logistic, y_kNN, poly_degree=2, save_path=False):
    X_train_poly = PolynomialFeatures(degree=poly_degree).fit_transform(X_train)
    X_test_poly = PolynomialFeatures(degree=poly_degree).fit_transform(X_test) 
    
    frequent_dummy = DummyClassifier(strategy='most_frequent').fit(X_train_poly, y_train)
    y_frequent = frequent_dummy.predict(X_test_poly)

    uniform_dummy = DummyClassifier(strategy='uniform').fit(X_train_poly, y_train)
    y_uniform = uniform_dummy.predict(X_test_poly)

    print('====== \n Results for dummy classifier (most frequent):')
    print(confusion_matrix(y_test, y_frequent))
    print(classification_report(y_test, y_frequent))

    print('====== \n Results for dummy classifier (uniform):')
    print(confusion_matrix(y_test, y_uniform))
    print(classification_report(y_test, y_uniform))

    print('====== \n Results for logistic regression:')
    print(confusion_matrix(y_test, ypreds))
    print(classification_report(y_test, ypreds))

    print('====== \n Results for kNN classifier:')
    print(confusion_matrix(y_test, kNN_preds))
    print(classification_report(y_test, kNN_preds))

    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_model.decision_function(X_test_poly))
    kNN_fpr, kNN_tpr, _ = roc_curve(y_test, kNN_model.predict_proba(X_test)[:,1])

    frequent_fpr, frequent_tpr, _ = roc_curve(y_test, frequent_dummy.predict_proba(X_test)[:,1])
    uniform_fpr, uniform_tpr, _ = roc_curve(y_test, uniform_dummy.predict_proba(X_test)[:,1])

    plt.plot(logistic_fpr,logistic_tpr)
    plt.plot(kNN_fpr, kNN_tpr, linewidth=1)
    plt.plot(frequent_fpr, frequent_tpr, linestyle='--')
    plt.plot(uniform_fpr, uniform_tpr, linestyle='-.', color='yellow', linewidth=1)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(['ROC for logistic regression','ROC for kNN ', 'dummy (most frequent) ROC', 'dummy (uniform) ROC'])
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    for f in os.listdir('.'):
        if '.txt' in f:
            title = f.split('_')[1].split('.')[0] 

            #(i) dataset id : 25-50-25-0  
            data = pd.read_csv(f'./{f}', comment='#', sep=',', header=None)
            data.columns = ['X1','X2','label']
            X, y = split_vis_data(data, save_path=f'./{title}_datavis.pdf')

            X_train, X_test, y_train, y_test = train_test_split(X,y)

            # (a)
            polyRange = range(1,7)  
            polynomial_CV_results = CV_polynomial(X_train, y_train, polyRange, LogisticRegression(penalty='l2', solver='lbfgs', C=1))

            plot_CV(polynomial_CV_results, 
                xlabel='order of polynomial', 
                title='cross-val results: PolynomialFeatures',
                save_path=f'./{title}_crossVal_q.pdf'
            )
            # from the plots, q=2 was chosen

            X_train_poly = PolynomialFeatures(degree=2).fit_transform(X_train)

            C_Range = np.geomspace(0.01,10)
            C_CV_results = CV_C(X_train_poly, y_train, C_Range)
            plot_CV(C_CV_results, 
                xlabel='C', 
                title='cross-val results: LogisticRegression C (wide range)',
                save_path=f'./{title}_crossVal_C_wide.pdf'
            )

            C_Range = np.linspace(1,2)
            C_CV_results = CV_C(X_train_poly, y_train, C_Range)
            plot_CV(C_CV_results, 
                xlabel='C', 
                title='cross-val results: LogisticRegression C (narrow range)',
                save_path=f'./{title}_crossVal_C_narrow.pdf'
            )
            # from the plots, C=1.6 was chosen

            logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.7)
            
            model = logistic_model.fit(X_train_poly, y_train)
            
            X_test_poly = PolynomialFeatures(degree=2).fit_transform(X_test)
            ypreds = model.predict(X_test_poly)
            plot_train_predict(data, get_pred_df(X_test, ypreds), save_path=f'{title}_logistic_datapreds.pdf')

            # (b) perform kNN Classification
            k_range = np.linspace(1,15)
            kNN_CV_results = CV_kNN(X_train, y_train, k_range)
            plot_CV(kNN_CV_results, 
                xlabel='number of neighbours (k)', 
                title='cross-val results: kNN K hyperparameter (wide range)',
                save_path = f'{title}_kNN_CrossVal_wide.pdf'
            )

            k_range = np.linspace(8,13)
            kNN_CV_results = CV_kNN(X_train, y_train, k_range)
            plot_CV(kNN_CV_results, 
                xlabel='number of neighbours (k)', 
                title='cross-val results: kNN K hyperparameter (narrow range)',
                save_path = f'{title}_kNN_CrossVal_narrow.pdf'
            )
            # from the plot, choose k=10
            kNN_model = KNeighborsClassifier(n_neighbors=10, weights='distance')

            kNN_model.fit(X_train_poly, y_train)
            kNN_preds_poly = kNN_model.predict(X_test_poly)
            plot_train_predict(data, get_pred_df(X_test,kNN_preds_poly), save_path=f'{title}_kNN_datavis_poly.pdf')

            kNN_model.fit(X_train, y_train)
            kNN_preds = kNN_model.predict(X_test)
            plot_train_predict(data, get_pred_df(X_test,kNN_preds), save_path=f'{title}_kNN_datavis.pdf')

            # (c) calculate confusion matrices for these models and for dummy classifier
            evaluate(X_train, y_train, X_test, y_test, ypreds, kNN_preds, save_path=f'{title}_ROC.pdf')
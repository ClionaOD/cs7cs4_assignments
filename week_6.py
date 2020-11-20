import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week5.php'
dataset = requests.get(dataset_url)
with open('./week_6_dataset.txt','w') as f:
    f.write(dataset.text)"""

# dataset id:20-200--20

def CV_kNregress(X, y, iter_list):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)

    for i in iter_list:
        
        def gaussian_kernel(distances):
            weights = np.exp(-i*(distances**2))
            return weights/np.sum(weights)
        
        kf = KFold(n_splits=5)
        errors = []
        for train, test in kf.split(X):
            model = KNeighborsRegressor(n_neighbors=len(X[train]), weights=gaussian_kernel).fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',i] = mean_err
        cv_df.loc['variance',i] = std

    return cv_df

def CV_gamma_ridge(X, y, iter_list, C):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    for i in iter_list:
        
        kf = KFold(n_splits=5)
        errors = []
        for train, test in kf.split(X):
            model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=i).fit(X[train], y[train])
            ypred = model.predict(X[test])
            
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',i] = mean_err
        cv_df.loc['variance',i] = std

    return cv_df

def CV_alpha_ridge(X, y, iter_list, gamma):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    for i in iter_list:
        
        kf = KFold(n_splits=5)
        errors = []
        for train, test in kf.split(X):
            model = KernelRidge(alpha=1.0/i, kernel='rbf', gamma=gamma).fit(X[train], y[train])
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

def CV_simult_gamma_C(X,y, g_range, save_path=False):
    cv_gamma_point1 = CV_gamma_ridge(X,y,g_range, C=0.1)
    cv_gamma_1 = CV_gamma_ridge(X,y,g_range, C=1)
    cv_gamma_100 = CV_gamma_ridge(X,y,g_range, C=100)
    
    fig, ax = plt.subplots()
    ax.errorbar(
        np.array(cv_gamma_point1.columns), 
        np.array(cv_gamma_point1.loc['mean error']), 
        yerr=cv_gamma_point1.loc['variance']
    )
    ax.errorbar(
        np.array(cv_gamma_1.columns), 
        np.array(cv_gamma_1.loc['mean error']), 
        yerr=cv_gamma_1.loc['variance']
    )
    ax.errorbar(
        np.array(cv_gamma_100.columns), 
        np.array(cv_gamma_100.loc['mean error']), 
        yerr=cv_gamma_100.loc['variance']
    )
    ax.set_xlabel('γ')
    ax.set_ylabel('mean squared error')
    ax.legend(['C=0.01','C=1','C=100'])
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def CV_simult_C_g(X, y, C_range, save_path=False):
    cv_C_1 = CV_alpha_ridge(X,y,C_range, gamma=0)
    cv_C_2 = CV_alpha_ridge(X,y,C_range, gamma=10)
    cv_C_3 = CV_alpha_ridge(X,y,C_range, gamma=25)
    
    fig, ax = plt.subplots()
    ax.errorbar(
        np.array(cv_C_1.columns), 
        np.array(cv_C_1.loc['mean error']), 
        yerr=cv_C_1.loc['variance']
    )
    ax.errorbar(
        np.array(cv_C_2.columns), 
        np.array(cv_C_2.loc['mean error']), 
        yerr=cv_C_2.loc['variance']
    )
    ax.errorbar(
        np.array(cv_C_3.columns), 
        np.array(cv_C_3.loc['mean error']), 
        yerr=cv_C_3.loc['variance']
    )
    ax.set_xlabel('C')
    ax.set_ylabel('mean squared error')
    ax.legend(['γ=0','γ=10','γ=25'])
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def perform_modelling(Xtrain, ytrain, Xtest, gammas, Cs):
    # set up figures
    fig1, ax1 = plt.subplots(nrows=5, sharex=True, sharey=True, figsize=[8.27,11.69])
    fig2, ax2 = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True, figsize=[8.27,11.69])

    cols=[f'C={C}' for C in Cs]
    rows=[f'γ={g}' for g in gammas]
    for ax, col in zip(ax2[0], cols):
        ax.set_title(col)
    for ax, row in zip(ax2[:,0], rows):
        ax.set_ylabel(row)

    # perform modelling over a range of γ and C
    for i, g in enumerate(gammas):
        
        def gaussian_kernel(distances):
            weights = np.exp(-g*(distances**2))
            return weights/np.sum(weights)

        model = KNeighborsRegressor(n_neighbors=len(Xtrain), weights=gaussian_kernel).fit(Xtrain, ytrain)

        ypred = model.predict(Xtest)

        ax1[i].scatter(Xtrain,ytrain,color='red',marker='+')
        ax1[i].plot(Xtest,ypred,color='green')
        ax1[i].set_title(f'γ={g}')
        
        fig1.text(0.5, 0.04, 'input x', ha='center')
        fig1.text(0.04, 0.5, 'output y', va='center', rotation='vertical')
        
        for j, C in enumerate(Cs):
            model = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=g).fit(Xtrain, ytrain)
            ypred = model.predict(Xtest)

            #print(f'==== \n dual_coef for g={g} and C={C} \n {model.dual_coef_}')

            ax2[i][j].scatter(Xtrain,ytrain,color='red',marker='+')
            ax2[i][j].plot(Xtest,ypred,color='green')
            fig2.text(0.5, 0.04, 'input x', ha='center')
            fig2.text(0.04, 0.5, 'output y', va='center', rotation='vertical')
    
    return fig1, fig2

if __name__ == "__main__":

    # (a) define dummy data
    dummyX = np.array([[-1,0],[0,1],[1,0]])
    dummy_X_train = dummyX[:,0]
    dummy_X_train = dummy_X_train.reshape(-1,1)
    dummy_y_train = dummyX[:,1]
    dummy_y_train = dummy_y_train.reshape(-1,1)

    dummy_X_test = np.array(np.linspace(-3,3)).reshape(-1,1)

    # set hyperparameters to test
    gammas = [0,1,5,10,25]
    Cs = [0.1,1,1000]
    
    # run modelling, plot figures
    dummy_fig1, dummy_fig2 = perform_modelling(dummy_X_train, dummy_y_train, dummy_X_test, gammas, Cs)
    #dummy_fig1.savefig('./fig1a.pdf')
    #dummy_fig2.savefig('./fig1c.pdf')
    plt.show()
    plt.close()

    #(b) repeat on real data
    dataset = pd.read_csv('./week_6_dataset.txt',sep=',',comment='#', header=None)
    X = dataset.iloc[:,0].values.reshape(-1,1)
    y = dataset.iloc[:,1].values.reshape(-1,1)

    plt.scatter(X,y, color='red', marker='+')
    plt.xlabel('input X')
    plt.ylabel('target y')
    #plt.savefig('./data_vs.pdf')
    plt.show()
    plt.close()

    #true train/test splits for cross validation
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y)

    #generate grid of values for predictions
        #Xtrain.max() == 1.0
        #Xtrain.min() == 1.0
        #therefore predictions for range(-3,3), use dummy_X_test from above

    # run modelling
    real_fig1, real_fig2 = perform_modelling(Xtrain, ytrain, dummy_X_test, gammas, Cs)
    #real_fig1.savefig('./fig2a.pdf')
    #real_fig2.savefig('./fig2c.pdf')
    plt.show()
    plt.close()

    # do cross validation
    g_wide = np.geomspace(1,100)
    cv_gamma = CV_kNregress(Xtrain,ytrain,g_wide)
    plot_CV(cv_gamma, 
        xlabel='γ', 
        title='cross-val results: KNeighborsRegressor γ (wide range)',
        save_path='./KNN_regress_gamma(wide)_CV.pdf'
    )
    g_narrow = np.linspace(25,50)
    cv_gamma = CV_kNregress(Xtrain,ytrain,g_narrow)
    plot_CV(cv_gamma, 
        xlabel='γ', 
        title='cross-val results: KNeighborsRegressor γ (narrow range)',
        save_path='./KNN_regress_gamma(narrow)_CV.pdf'
    )
    # choose γ=50

    g_wide = np.geomspace(1,100)
    CV_simult_gamma_C(Xtrain, ytrain, g_wide, save_path='./CV_gamma(wide)_varC.pdf')

    g_narrow = np.linspace(0,25)
    CV_simult_gamma_C(Xtrain, ytrain, g_narrow, save_path='./CV_gamma(narrow)_varC.pdf')

    C_wide = np.geomspace(0.01,100)
    CV_simult_C_g(Xtrain, ytrain, C_wide, save_path='./CV_C(wide)_vargamma.pdf')

    # choose γ=10, evaluate C over narrower range
    cv_C = CV_alpha_ridge(Xtrain, ytrain, iter_list=np.linspace(1,20), gamma=10)
    plot_CV(cv_C, 
        xlabel='C', 
        title='cross-val results: KernelRidge C (narrow range), γ=10', 
        save_path='./CV_C(narrow)_vargamma.pdf'
    )

    # choose γ=10, C=5

    g1=10
    C=5
    model1 = KernelRidge(alpha=1.0/C, kernel='rbf', gamma=g1).fit(Xtrain, ytrain)
    preds1 = model1.predict(Xtest)

    g2=50
    def gaussian_kernel(distances):
        weights = np.exp(-g2*(distances**2))
        return weights/np.sum(weights)

    model2 = KNeighborsRegressor(n_neighbors=len(Xtrain), weights=gaussian_kernel).fit(Xtrain, ytrain)
    preds2 = model2.predict(Xtest)

    fig, ax = plt.subplots(ncols=2, sharey=True, sharex=True)
    ax[0].scatter(Xtrain,ytrain,color='red',marker='+')
    ax[1].scatter(Xtrain,ytrain,color='red',marker='+')
    ax[0].scatter(Xtest,preds1,color='green')
    ax[1].scatter(Xtest,preds1,color='green')
    ax[0].set_title('Kernalised Ridge Regression C=5, γ=10')
    ax[1].set_title('Gaussian Weights kNN, γ=50')
    fig.text(0.5, 0.04, 'input x', ha='center')
    fig.text(0.04, 0.5, 'output y', va='center', rotation='vertical')
    plt.savefig('./model predictions.pdf')
    plt.show()
    plt.close()

    baseline = DummyRegressor(strategy='mean').fit(Xtrain,ytrain)
    y_baseline = baseline.predict(Xtest)
    
    MSE1 = mean_squared_error(ytest, preds1)
    print(f'Mean square error for kNN with Gaussian Kernel = {MSE1}')

    MSE2 = mean_squared_error(ytest, preds2)
    print(f'Mean square error for kernalised Ridge = {MSE2}')

    MSEDummy = mean_squared_error(ytest, y_baseline)
    print(f'Mean square error for kernalised Ridge = {MSEDummy}')


    









import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week3.php'
dataset = requests.get(dataset_url)
with open('./week_3_dataset.txt','w') as f:
    f.write(dataset.text)"""

# NOTE: plotting code has been ommitted for clarity

#(i) (a)
# dataset id : 9--18-9
df = pd.read_csv('./week_3_dataset.txt', sep=',', comment='#', header=None)
df.columns = ['X1','X2','target']
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

# (i)(b) 
Xpoly = PolynomialFeatures(degree=5).fit_transform(X)
# X.shape = (199,2)
# Xpoly.shape = (199,21)

alphas = [.0001,.001,1]

#(i)(c)
#generate grid of feature values
    #range of X is (-1,1) therefore extend grid beyond this.
Xtest = []
grid = np.linspace(-2,2)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
Xtest_poly = PolynomialFeatures(degree=5).fit_transform(Xtest)

for a in alphas:
    lasso = linear_model.Lasso(alpha=a).fit(Xpoly,y)
    print(f'====== \n Lasso parameters for alpha={a}: \n intercept: {lasso.intercept_} \n slope: {lasso.coef_}')

    lasso_preds = lasso.predict(Xtest_poly)

    #(i)(e)
    ridge = linear_model.Ridge(alpha=a).fit(Xpoly,y)
    print(f'====== \n Ridge parameters for alpha={a}: \n intercept: {ridge.intercept_} \n slope: {ridge.coef_}')

    ridge_preds = ridge.predict(Xtest_poly)

#(ii)
#(a)
a = 0.5 #C=1 therefore alpha=1/2
lasso = linear_model.Lasso(alpha=a)

folds = [5, 2, 10, 25, 50, 100]
cv_df_k = pd.DataFrame(index=['mean error', 'variance'], columns=folds)

for k in folds:
    kf = KFold(n_splits=k)

    errors = []
    for i in range(5):
        for train, test in kf.split(Xpoly):
            model =  lasso.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            
            error = mean_squared_error(y[test],ypred)
            errors.append(error)
    
    mean_err = np.mean(np.array(errors))
    std = np.std(np.array(errors))
    
    cv_df_k.loc['mean error',k] = mean_err
    cv_df_k.loc['variance',k] = std
    print(f'===== \n {k}-fold Cross-Val results: \n mean error = {mean_err} \n standard deviation = {std}')

# (ii) (b) - (d)
def cross_val_C(Xpoly, y, iter_list, regularisation):
    cv_df = pd.DataFrame(index=['mean error', 'variance'], columns=iter_list)
    
    for a in iter_list:
        if regularisation == 'lasso':
            model = linear_model.Lasso(alpha=a)
        elif regularisation == 'ridge':
            model = linear_model.Ridge(alpha=a)
        
        kf = KFold(n_splits=5)

        errors = []
        for i in range(5):
            for train, test in kf.split(Xpoly):
                model =  model.fit(Xpoly[train], y[train])
                ypred = model.predict(Xpoly[test])
                
                error = mean_squared_error(y[test],ypred)
                errors.append(error)
        
        mean_err = np.mean(np.array(errors))
        std = np.std(np.array(errors))
        
        cv_df.loc['mean error',a] = mean_err
        cv_df.loc['variance',a] = std

    return cv_df

# first, examine a wide range that is logarithmically spaced
a_lasso_wide = cross_val_C(Xpoly, y, np.geomspace(0.0001,1), 'lasso')

# based on this plot, narrow the range 
a_lasso_narrow = cross_val_C(Xpoly, y, np.linspace(0.0001,0.1), 'lasso')
#We get a alpha = 0.01 as a sensible choice, from the cross validation plot (see report).

# repeat for Ridge regression
    # use a different range, based on results from part (i)
ridge_range = np.geomspace(0.01,100)
a_ridge_wide = cross_val_C(Xpoly, y, ridge_range, 'ridge')

# narrow the range tested after initial inspection
ridge_range = np.linspace(0.01,10)
a_ridge_narrow = cross_val_C(Xpoly, y, ridge_range, 'ridge')
# from the plot, we find that a good choice for alpha is 1 (see report).



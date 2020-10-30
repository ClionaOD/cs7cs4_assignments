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

# dataset id : 9--18-9
df = pd.read_csv('./week_3_dataset.txt', sep=',', comment='#', header=None)
df.columns = ['X1','X2','target']
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X1,X2,y, color='r')
ax.set_xlabel('input 1')
ax.set_ylabel('input 2')
ax.set_zlabel('target value')
ax.view_init(elev=15., azim=-55.)
#plt.savefig('./data_vis.pdf')
plt.close()

# (i)(b) 
Xpoly = PolynomialFeatures(degree=5).fit_transform(X)
# X.shape = (199,2)
# Xpoly.shape = (199,21)

alphas = [.0001,.001,1]

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

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1],lasso_preds, color='palegoldenrod',label='predictions')
    surf._edgecolors2d = surf._edgecolors3d
    surf._facecolors2d = surf._facecolors3d
    
    scat = ax.scatter(X1,X2,y, color='r',label='training points')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')

    ax.view_init(elev=15., azim=-55.)

    ax.set_title(f'Lasso: 1/2C = {a}')
    ax.legend()
    
    plt.tight_layout()
    save_path = f'./Lasso_plot_alpha={a}.pdf'
    #plt.savefig(save_path)
    #plt.show()
    plt.close()

    ridge = linear_model.Ridge(alpha=a).fit(Xpoly,y)
    print(f'====== \n Ridge parameters for alpha={a}: \n intercept: {ridge.intercept_} \n slope: {ridge.coef_}')

    ridge_preds = ridge.predict(Xtest_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1],ridge_preds, color='palegoldenrod',label='predictions')
    surf._edgecolors2d = surf._edgecolors3d
    surf._facecolors2d = surf._facecolors3d

    scat = ax.scatter(X1,X2,y, color='r',label='training points')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')

    ax.view_init(elev=15., azim=-55.)

    ax.set_title(f'Ridge: 1/2C = {a}')
    ax.legend()

    plt.tight_layout()
    save_path = f'./Ridge_plot_alpha={a}.pdf'
    #plt.savefig(save_path)
    #plt.show()
    plt.close()

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

fig, ax = plt.subplots()
ax.errorbar(np.array(cv_df_k.columns), np.array(cv_df_k.loc['mean error']), yerr=cv_df_k.loc['variance'])
ax.set_xlabel('number of folds (k)')
ax.set_ylabel('mean prediction error')
#plt.savefig('./kfold_mean_std.pdf')
#plt.show()
plt.close()

# (b) - (d)
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

a_lasso_wide = cross_val_C(Xpoly, y, np.geomspace(0.0001,1), 'lasso')

fig, ax = plt.subplots()
ax.errorbar(np.array(a_lasso_wide.columns), np.array(a_lasso_wide.loc['mean error']), yerr=a_lasso_wide.loc['variance'])
ax.set_xlabel('value for alpha (1/2C)')
ax.set_ylabel('mean prediction error')
ax.set_title('lasso cross val for wide range of alpha')
#plt.savefig('./cv_lasso_1.pdf')
#plt.show()
plt.close()

# based on this plot, narrow the range 
a_lasso_narrow = cross_val_C(Xpoly, y, np.linspace(0.0001,0.1), 'lasso')

fig, ax = plt.subplots()
ax.errorbar(np.array(a_lasso_narrow.columns), np.array(a_lasso_narrow.loc['mean error']), yerr=a_lasso_narrow.loc['variance'])
ax.set_xlabel('value for alpha (1/2C)')
ax.set_ylabel('mean prediction error')
ax.set_title('lasso cross val for smaller range of alpha')
#plt.savefig('./cv_lasso_2.pdf')
#plt.show()
plt.close()
#We get a alpha = 0.01 as a sensible choice, from the cross validation plot.

# repeat for Ridge regression
    # use a different range, based on results from part (i)
ridge_range = np.geomspace(0.01,100)
a_ridge_wide = cross_val_C(Xpoly, y, ridge_range, 'ridge')

fig, ax = plt.subplots()
ax.errorbar(np.array(a_ridge_wide.columns), np.array(a_ridge_wide.loc['mean error']), yerr=a_ridge_wide.loc['variance'])
ax.set_xlabel('value for alpha (1/2C)')
ax.set_ylabel('mean prediction error')
ax.set_title('ridge cross val over wide range of alpha')
#plt.savefig('./cv_ridge_1.pdf')
#plt.show()
plt.close()

# narrow the range tested after initial inspection
ridge_range = np.linspace(0.01,10)
a_ridge_narrow = cross_val_C(Xpoly, y, ridge_range, 'ridge')

fig, ax = plt.subplots()
ax.errorbar(np.array(a_ridge_narrow.columns), np.array(a_ridge_narrow.loc['mean error']), yerr=a_ridge_narrow.loc['variance'])
ax.set_xlabel('value for alpha (1/2C)')
ax.set_ylabel('mean prediction error')
ax.set_title('ridge cross val over narrow range of alpha')
#plt.savefig('./cv_ridge_1.pdf')
#plt.show()
plt.close()



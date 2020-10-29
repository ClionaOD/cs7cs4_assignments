import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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
plt.savefig('./data_vis.pdf')
plt.close()

# (i)(b) 
Xpoly = PolynomialFeatures(degree=5).fit_transform(X)
# X.shape = (199,2)
# Xpoly.shape = (199,21)

Cs = [.0001,.001,1]

#generate grid of feature values
    #range of X is (-1,1) therefore extend grid beyond this.
Xtest = []
grid = np.linspace(-2,2)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
Xtest_poly = PolynomialFeatures(degree=5).fit_transform(Xtest)

for C in Cs:
    lasso = linear_model.Lasso(alpha=C).fit(Xpoly,y)
    print(f'====== \n parameters for C={C}: \n intercept: {lasso.intercept_} \n slope: {lasso.coef_}')

    preds = lasso.predict(Xtest_poly)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    surf = ax.plot_trisurf(Xtest[:,0],Xtest[:,1],preds, color='palegoldenrod',label='predictions')
    surf._edgecolors2d = surf._edgecolors3d
    surf._facecolors2d = surf._facecolors3d
    
    scat = ax.scatter(X1,X2,y, color='r',label='training points')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')

    ax.view_init(elev=15., azim=-55.)

    ax.set_title(f'C = {C}')
    ax.legend()
    
    save_path = f'./Lasso_plot_C={C}.pdf'
    plt.savefig(save_path)
    plt.show()
    plt.close()
    






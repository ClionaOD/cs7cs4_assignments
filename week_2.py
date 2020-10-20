import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week2.php'
dataset = requests.get(dataset_url)
with open('./week_2_dataset.txt','w') as f:
    f.write(dataset.text)"""

# dataset id : 19--38-19
df = pd.read_csv('./week_2_dataset.txt', sep=',', comment='#', header=None)
df.columns = ['X1','X2','label']
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

df_pos = df[df['label'] == 1]
df_neg = df[df['label'] == -1]

plt.scatter(df_pos.iloc[:,0],df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0],df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['target = +1','target = -1'])
plt.savefig('./data_visualisation.pdf')
plt.show()
plt.close()

model = LogisticRegression(penalty='none', solver='lbfgs')
clf = model.fit(X, y)
print(f'intercept is : {model.intercept_} \n slope is : {model.coef_}')

# get decision boundary line by rearranging thetaTx = 0
# theta0 = clf.intercept_
# theta1 = clf.coef_[0][0]
# theta2 = clf.coef_[0][1]
# theta0*x0 + theta1*x1 + theta2*x2 = 0
# x2 = -theta0(1) - theta1*x1

boundary_intercept = -clf.intercept_ / clf.coef_[0][1]
boundary_slope = -clf.coef_[0][0] / clf.coef_[0][1]
boundary_x = X1
boundary_y = boundary_slope*boundary_x + boundary_intercept

y_pred = clf.predict(X)
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_pred
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]

fig = plt.figure()
plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.plot(boundary_x, boundary_y, linewidth=2)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['decision boundary','target = +1','target = -1', 'predicted = +1', 'predicted = -1'])
plt.show()
plt.savefig('./vis_with_preds.pdf')
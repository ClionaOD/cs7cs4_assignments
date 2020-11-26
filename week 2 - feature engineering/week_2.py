import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
plt.legend(['target = +1','target = -1'], loc='lower right')
plt.savefig('./data_visualisation.pdf')
plt.close()

logistic_model = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
print(f'intercept is : {logistic_model.intercept_} \n slope is : {logistic_model.coef_}')

# get decision boundary line by rearranging thetaTx = 0
# theta0 = logistic_model.intercept_
# theta1 = logistic_model.coef_[0][0]
# theta2 = logistic_model.coef_[0][1]
# theta0*x0 + theta1*x1 + theta2*x2 = 0
# x2 = -theta0(1) - theta1*x1

boundary_intercept = -logistic_model.intercept_ / logistic_model.coef_[0][1]
boundary_slope = -logistic_model.coef_[0][0] / logistic_model.coef_[0][1]
boundary_x = X1
boundary_y = boundary_slope*boundary_x + boundary_intercept

y_pred = logistic_model.predict(X)
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_pred
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]

plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.plot(boundary_x, boundary_y, linewidth=2)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['decision boundary','target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
#plt.savefig('./vis_with_preds.pdf')
plt.close()

model1 = LinearSVC(C=0.001).fit(X,y)
print(f'for C = 0.001: \n intercept is : {model1.intercept_} \n slope is : {model1.coef_}')
y_1 = model1.predict(X)
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_1
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]
#calculate decision boundary
bound1_intercept = -model1.intercept_ / model1.coef_[0][1]
bound1_slope = -model1.coef_[0][0] / model1.coef_[0][1]
bound1_x = X1
bound1_y = bound1_slope*bound1_x + bound1_intercept
#plot training, predictions and decision boundary
plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.plot(bound1_x, bound1_y, linewidth=2)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['decision boundary (C=0.001)','target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
plt.savefig('./SVM_C=0.001.pdf')
#plt.show()
plt.close()

model2 = LinearSVC(C=1).fit(X,y)
print(f'for C = 1: \n intercept is : {model2.intercept_} \n slope is : {model2.coef_}')
y_2 = model2.predict(X)
#split up for plotting
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_2
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]
#calculate decision boundary
bound2_intercept = -model2.intercept_ / model2.coef_[0][1]
bound2_slope = -model2.coef_[0][0] / model2.coef_[0][1]
bound2_x = X1
bound2_y = bound2_slope*bound2_x + bound2_intercept
#plot training, predictions and decision boundary
plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.plot(bound2_x, bound2_y, linewidth=2)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['decision boundary (C=1)','target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
plt.savefig('./SVM_C=1.pdf')
#plt.show()
plt.close()

model3 = LinearSVC(C=1000).fit(X,y)
print(f'for C = 1000: \n intercept is : {model3.intercept_} \n slope is : {model3.coef_}')
y_3 = model3.predict(X)
#split up for plotting
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_3
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]
#calculate decision boundary
bound3_intercept = -model3.intercept_ / model3.coef_[0][1]
bound3_slope = -model3.coef_[0][0] / model3.coef_[0][1]
bound3_x = X1
bound3_y = bound3_slope*bound3_x + bound3_intercept
#plot training, predictions and decision boundary
plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.plot(bound3_x, bound3_y, linewidth=2)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['decision boundary (C=1000)','target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
#plt.savefig('./SVM_C=1000.pdf')
#plt.show()
plt.close()

# feature engineering 

X3 = df.iloc[:,0] ** 2
X4 = df.iloc[:,1] ** 2
X = np.column_stack((X1,X2,X3,X4))

logistic_model2 = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
print(f'intercept is : {logistic_model2.intercept_} \n slope is : {logistic_model2.coef_}')
y_feat_eng = logistic_model2.predict(X)
#split up for plotting
pred_df = pd.DataFrame(columns=['X1','X2','y_pred'])
pred_df['X1'] = X1
pred_df['X2'] = X2
pred_df['y_pred'] = y_feat_eng
pred_pos = pred_df[pred_df['y_pred'] == 1]
pred_neg = pred_df[pred_df['y_pred'] == -1]
#plot training, predictions and decision boundary
plt.scatter(df_pos.iloc[:,0], df_pos.iloc[:,1], marker='o', color='b', s=18)
plt.scatter(df_neg.iloc[:,0], df_neg.iloc[:,1], marker='+', color='purple', s=18)
plt.scatter(pred_pos.iloc[:,0], pred_pos.iloc[:,1], marker='h', color='darkorange', s=9)
plt.scatter(pred_neg.iloc[:,0], pred_neg.iloc[:,1], marker='v', color='yellow', s=9)
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.legend(['target = +1','target = -1', 'predicted = +1', 'predicted = -1'], loc='lower right')
plt.savefig('./feature_eng.pdf')
plt.show()
plt.close()
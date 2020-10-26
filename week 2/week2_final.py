import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from statistics import mode

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week2.php'
dataset = requests.get(dataset_url)
with open('./week_2_dataset.txt','w') as f:
    f.write(dataset.text)"""

# NOTE: code for plotting figures has been omitted for clarity

# dataset id : 19--38-19
df = pd.read_csv('./week_2_dataset.txt', sep=',', comment='#', header=None)
df.columns = ['X1','X2','label']
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X = np.column_stack((X1,X2))
y = df.iloc[:,2]

#(a)(ii) use sklearn to train a logistic regression model
logistic_model = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
print(f'==== \n logistic regression results: \n intercept is : {logistic_model.intercept_} \n feature parameters : {logistic_model.coef_}')

#(a)(iii) use the model to predict y from the training data
y_pred = logistic_model.predict(X)

# (a)(iii) plot the decision boundary
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

#(b)
# use sklearn to train a linear SVM for a wide range of values for C
# C = 0.001
# (i) train
model1 = LinearSVC(C=0.001).fit(X,y)
print(f' ==== \n SVM results for C = 0.001: \n intercept is : {model1.intercept_} \n feature parameters : {model1.coef_}')
# (ii) predict
y_1 = model1.predict(X)
# (ii) calculate decision boundary
bound1_intercept = -model1.intercept_ / model1.coef_[0][1]
bound1_slope = -model1.coef_[0][0] / model1.coef_[0][1]
bound1_x = X1
bound1_y = bound1_slope*bound1_x + bound1_intercept

# C = 1
# (i) train
model2 = LinearSVC(C=1).fit(X,y)
print(f'==== \n SVM results for C = 1: \n intercept is : {model2.intercept_} \n feature parameters : {model2.coef_}')
# (ii) predict
y_2 = model2.predict(X)
# (ii) calculate decision boundary
bound2_intercept = -model2.intercept_ / model2.coef_[0][1]
bound2_slope = -model2.coef_[0][0] / model2.coef_[0][1]
bound2_x = X1
bound2_y = bound2_slope*bound2_x + bound2_intercept

# C = 1000
# (i) train
model3 = LinearSVC(C=1000).fit(X,y)
print(f'==== \n SVM results for C = 1000: \n intercept is : {model3.intercept_} \n feature parameters : {model3.coef_}')
# (ii) predict
y_3 = model3.predict(X)
# (ii) calculate decision boundary
bound3_intercept = -model3.intercept_ / model3.coef_[0][1]
bound3_slope = -model3.coef_[0][0] / model3.coef_[0][1]
bound3_x = X1
bound3_y = bound3_slope*bound3_x + bound3_intercept

# (c)
# feature engineering
# (i) create additional features by squaring X1 and X2 
X3 = df.iloc[:,0] ** 2
X4 = df.iloc[:,1] ** 2
X = np.column_stack((X1,X2,X3,X4))

# (i) train a logistic regression classifier
logistic_model2 = LogisticRegression(penalty='none', solver='lbfgs').fit(X, y)
print(f'==== \n logistic regression results after feature engineering \n intercept is : {logistic_model2.intercept_} \n feature parameters : {logistic_model2.coef_}')
# (ii) predict target values
y_feat_eng = logistic_model2.predict(X)

# (iii) compare the performance against a reasonable baseline predictor
    # set baseline as model that always predicts most common class

def baseline(X,y):
    most_common = mode(y)
    pred_df = pd.DataFrame(X)
    pred_df['preds'] = most_common
    return pred_df

base_df = baseline(X,y)
base_df['true'] = y

feat_eng_df = pd.DataFrame(X)
feat_eng_df['preds'] = y_feat_eng
feat_eng_df['true'] = y

performance_baseline = len(np.where(base_df['preds'] != base_df['true'])[0])
print(f'Baseline model number of incorrect predictions is {performance_baseline}')

engineered_baseline = len(np.where(feat_eng_df['preds'] != feat_eng_df['true'])[0])
print(f'Logistic regression model (with feature engineering) number of incorrect predictions is {engineered_baseline}')


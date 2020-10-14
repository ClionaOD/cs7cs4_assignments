import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# (i) read in the dataset downloaded id:7-4233.6-21 

"""dataset_url = 'https://www.scss.tcd.ie/Doug.Leith/CSU44061/week1.php'
dataset = requests.get(dataset_url)
with open('./dataset.txt','w') as f:
    f.write(dataset.text)"""    

df = pd.read_csv('./dataset.txt')
X = np.array(df.index)
X = X.reshape(-1,1)
y = np.array(df.iloc[:,0])
y = y.reshape(-1,1)

# visualise the data
plt.scatter(X,y)
plt.xlabel('input X')
plt.ylabel('output y')
#plt.savefig('./visual_raw.pdf')
plt.close()

# (ii) normalise the data
    # Xnorm = (X - Xmin) / (Xmax - Xmin)
Xnorm = (X - X.min()) / (X.max() - X.min())
ynorm = (y - y.min()) / (y.max() - y.min())
plt.scatter(Xnorm,ynorm)
plt.xlabel('input X (normalised)')
plt.ylabel('output y (normalised)')
#plt.savefig('./visual_norm.pdf')
plt.close()

# (iii) use gradient descent to train a linear regression model
    # feature x = Xnorm
    # linear model h(x) = theta0 + theta1(x)
    # parameters theta0 and theta1
    # cost function J(theta0, theta1) = see hand written
        # we are summing error (i.e. model output minus actual value squared) over all training points
        # m = len(df)

#choose starting point for paramaters in the normalised range
theta0 = 0.19
theta1 = 0.81
alpha = 0.01

features = list(zip(Xnorm,ynorm))

def linear_least_squares(features, theta0, theta1):
    # features is an array with m number of obserbations and x,y outputs
    # theta0 is parameter for y intercept, chosen from SGD
    # theta1 is parameter for slope, chosen from SGD
    m = len(features)
    sum_square_errors = 0
    for x,y in features:
        E = (theta0 - theta1*x) - y
        Esq = E**2
        sum_square_errors += Esq
    J = 1/m * sum_square_errors
    return J

def get_derivatives(features, theta0, theta1, alpha):
    m = len(features)
    
    deriv0 = 0
    for x,y in features:
        E = theta0 - theta1*x - y
        deriv0 += E
    deriv0 = ((-2 * alpha) / m) * deriv0
    
    deriv1 = 0
    for x,y in features:
        E = ((theta0 - theta1*x) - y) * x
        deriv1 += E
    deriv1 = ((-2 * alpha) / m) * deriv1

    return deriv0, deriv1

cost = pd.DataFrame(columns=['cost'])
for i in range(70):
    deriv0, deriv1 = get_derivatives(features, theta0, theta1, alpha)
    theta0 = theta0 + deriv0
    theta1 = theta1 + deriv1

    J = linear_least_squares(features, theta0, theta1)
    cost.loc[len(cost)] = J

# (b) (i) plot how the cost function changes over time
plt.scatter(cost.index,cost['cost'])
plt.xlabel('number of iterations')
plt.ylabel(f'J (cost function) - alpha = {alpha}')
#plt.savefig(f'./cost_function_alpha{alpha}.pdf')
plt.close()

# (b) (ii) report the dinal paramater values and cost function
print(f'final value for y intercept paramater : theta0={theta0} \n final value for slope paramater : theta1={theta1} \n cost function converged to {J}')

# (b) (iii) compare to a baseline that predicts a constant
theta0_baseline = 0.5
theta1_baseline = 0
J_baseline = linear_least_squares(features, theta0_baseline, theta1_baseline)
print(f' the baseline cost function value for theta0={theta0_baseline} and theta1={theta1_baseline} was {J_baseline}')

# (b) (iv) use scikit learn to train on the data and compare
model = LinearRegression().fit(X=Xnorm, y=ynorm)
print(f'scikit learn results: \n intercept theta0={model.intercept_} \n slope theta1={model.coef_}')
ypred_scikit = model.intercept_ + (model.coef_)*X
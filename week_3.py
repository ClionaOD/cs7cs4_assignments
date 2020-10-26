import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
ax.scatter(X1,X2,y)
ax.set_xlabel('input 1')
ax.set_ylabel('input 2')
ax.set_zlabel('target value')
plt.savefig('./data_vis.pdf')
plt.close()
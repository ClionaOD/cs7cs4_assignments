import requests
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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

plt.scatter(df_pos.iloc[:,0],df_pos.iloc[:,1], marker='o')
plt.scatter(df_neg.iloc[:,0],df_neg.iloc[:,1], marker='+')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.savefig('./data_visualisation.pdf')
plt.close()
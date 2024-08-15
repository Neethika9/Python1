#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install xgboost')


# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# In[71]:


#code to merge to data sets into single one
df1 = pd.read_csv(r"calories.csv") 
df2 = pd.read_csv(r"exercise.csv") 
  
# print the datasets 
print(df1.head()) 
print(df2.head())
merged_df = pd.merge(df1, df2, how='outer') 
merged_df.to_csv('calories.csv', index=False)


# In[72]:


df = pd.read_csv('calories.csv')
df.head()


# In[74]:


df.shape


# In[54]:


df.info()


# In[75]:


df2.describe()


# In[77]:


sb.scatterplot(df['Height'], df['Weight'])
plt.show()


# In[78]:


sb.scatterplot(df['Weight'], df['Age'])
plt.show()


# In[79]:


features = ['Age', 'Height', 'Weight', 'Duration']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x[col], x['Calories'])
plt.tight_layout()
plt.show()


# In[81]:


features = df.select_dtypes(include='float').columns

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()


# In[82]:


#binary conversion
df.replace({'male': 0, 'female': 1},
           inplace=True)
df.head()


# In[83]:


#heatmap
plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.9,
           annot=True,
           cbar=False)
plt.show()


# In[86]:


#splitting into traing and testing
features = df.drop(['User_ID', 'Calories'], axis=1)
target = df['Calories'].values

X_train, X_val,    Y_train, Y_val = train_test_split(features, target,
                                      test_size=0.1,
                                      random_state=22)
X_train.shape, X_val.shape


# In[87]:


#normalization 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[88]:


from sklearn.metrics import mean_absolute_error as mae
models = [LinearRegression(), XGBRegressor(),
          Lasso(), RandomForestRegressor(), Ridge()] # l1 and l2 regularizations as lasso and ridge functions.

for i in range(5):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')

    train_preds = models[i].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    val_preds = models[i].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()


# In[ ]:





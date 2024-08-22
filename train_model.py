#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# In[2]:


df = pd.read_csv('C:\\Users\\gkais\\Documents\\National University\\Masters_Data_Science\\ANA680\\Flask Projects\\Week 2 Midterm\\StudentsPerformance.csv')


# In[3]:


df.head()


# In[4]:


X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']


# In[5]:


y = y.astype('category').cat.codes


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[8]:


joblib.dump(model, 'model.pkl')


# In[9]:


y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


# In[ ]:





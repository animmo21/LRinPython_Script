#!/usr/bin/env python
# coding: utf-8

# # Linear Modelling in Python

# In[1]:


import pandas as pd 


# In[2]:


df = pd.read_csv('regrex1 2.csv')


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


df.head()


# In[5]:


df['x'].head()


# In[6]:


df['y'].head()


# In[7]:


plt.scatter(df['x'], df['y'])
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[8]:


get_ipython().system('pip install scikit-learn')


# In[9]:


import numpy as np 
from sklearn.linear_model import LinearRegression


# In[10]:


x = np.array(df['x']).reshape((-1,1))
y = np.array(df['y'])


# In[11]:


model = LinearRegression()


# In[12]:


model.fit(x,y)


# In[13]:


intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)


# In[14]:


print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")


# In[15]:


y_pred = model.predict(x)


# In[16]:


y_pred


# In[17]:


plt.plot(df['x'], y_pred)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[18]:


plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[ ]:





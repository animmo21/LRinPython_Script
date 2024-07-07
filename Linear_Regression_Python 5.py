#!/usr/bin/env python
# coding: utf-8

# # Linear Modelling in Python

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression
import sys

print("Linear Modeling in Python")

# In[2]:

datafile = sys.argv[1]
df = pd.read_csv(datafile)


# In[3]:


df.head()


# In[4]:


df['x'].head()


# In[5]:


df['y'].head()


# In[6]:


plt.scatter(df['x'], df['y'])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.savefig("Py_orig.png")


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
plt.savefig("Py_lm.png")


# In[ ]:





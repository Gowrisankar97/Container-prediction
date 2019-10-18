#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#importing file
d=pd.read_excel("container_1.xls")


# In[3]:


#shape eof thta input file
d.shape


# In[4]:


d.head(5)


# In[5]:


#number of column sin dataset
d.columns


# In[6]:


#dropping  columns which are ar ehaving hifh correlation 
h=d.drop(["Ship_Name","Ship_Type","Ship_Type_Grouping","Expr1008","Expr1009","IMO_No","Classed_By","Date_Acquired"],axis=1)


# In[7]:


c=h.dropna(axis=0,how="any")


# In[8]:


c.isnull().sum()


# In[9]:


Y=c["TEU"]


# In[10]:


X=c.drop(["TEU","Displacement_Tonnage","Engine_HP_Total","LBP","Gt","Engine_RPM"],axis=1)


# In[11]:


X.head(10)


# In[14]:


#visualising pair plot
sns.pairplot(X)


# In[15]:


#importing linear regression model
from sklearn.linear_model import LinearRegression


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


#seprating train and test dataset
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)


# In[18]:


model = LinearRegression()
model.fit(xtrain, ytrain)


# In[19]:


#test data score
model.score(xtest,ytest)


# In[20]:


#intercept
model.coef_


# In[21]:


#prediction 
#ENTER
#(Dwt	LOA	Beam	Draft	Depth	Speed) 


# In[22]:


model.predict([[46000,228,32,12.5,20,20]])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing libraries#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


#importing the dataset#
Dataset = pd.read_csv('student_scores.csv.txt')
x = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:,-1].values


# In[16]:


Dataset.head()


# In[20]:


#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


# In[21]:


#Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[22]:


#Predicting the Test set results
y_pred = regressor.predict(x_test)


# In[25]:


#Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('No of study Hours vs Student scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[26]:


#Visualising the Training set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('No of study Hours vs Student scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[30]:


#What will be predicted score if a student studies for 9.25 hrs/ day?
y_pred = regressor.predict([[9.25]])
y_pred


# In[ ]:





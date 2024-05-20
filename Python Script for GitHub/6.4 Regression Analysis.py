#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing Libraries & Data

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# This option ensures that the graphs you create are displayed within the notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# import datasets

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath = os.path.join(path, '.pkl', 'all_merge_numbers2.pkl')

# assign df name
df = pd.read_pickle(filepath)


# ### 2. Data Cleaning

# In[6]:


# remove column limit of output 
pd.options.display.max_columns = None

# remove row limit of output 
pd.options.display.max_rows = None


# In[9]:


df.head()


# In[57]:


# choosing select columns for regression

reg = df[['Entity', 'Anxiety', 'Depressive', 'Social_support',]]


# In[58]:


reg.head()


# In[59]:


reg.shape


# In[60]:


# checking for missing values

reg.isnull().sum()


# In[161]:


# below, it said NaN values cannot be present for regressions so I came back up here to drop the NaN values

reg.dropna(inplace = True)


# In[62]:


# checking for missing values

reg.isnull().sum()


# In[63]:


# checking for duplicates

dups = reg.duplicated()


# In[64]:


dups.shape
# no dups


# In[65]:


# checking for extreme values
# histogram of anxiety 

sns.histplot(reg['Anxiety'], bins=25, kde = True)


# In[66]:


# checking for extreme values
# histogram of depressive 

sns.histplot(reg['Depressive'], bins=25, kde = True)


# In[162]:


# checking for extreme values
# histogram of social_support

sns.histplot(reg['Social_support'], bins=20, kde = True)


# ### 3. Data Preparation for Regression

# HYPOTHESIS: 
# 
# - The higher the social support, the higher the prevalence of anxiety disorder.
# - The higher the social support, the lower the prevalence of depressive disorder.

# #### a. Anxiety Disorder

# In[163]:


# creating a scatterplot using matplotlib to see relationship

reg.plot(x = 'Social_support', y='Anxiety',style='o') 
plt.title('Social Support vs Prevalence of Anxiety Disorder')  
plt.xlabel('Social support')  
plt.ylabel('Prevalence of anxiety disorder')  
plt.show()


# In[120]:


# reshaping the variables into NumPy arrays and putting them into separate objects

X = reg['Social_support'].values.reshape(-1,1)
y_anxiety = reg['Anxiety'].values.reshape(-1,1)


# In[164]:


# splitting the data into training set and testing set

X_train, X_test, y_anxiety_train, y_anxiety_test = train_test_split(X, y_anxiety, test_size=0.3, random_state=0)


# #### b. Depressive Disorder

# In[115]:


# creating a scatterplot using matplotlib to see relationship

reg.plot(x = 'Social_support', y='Depressive',style='o') 
plt.title('Social Support vs Prevalence of Depressive Disorder')  
plt.xlabel('Social support')  
plt.ylabel('Prevalence of depressive disorder')  
plt.show()


# In[136]:


# reshaping the variables into NumPy arrays and putting them into separate objects

X = reg['Social_support'].values.reshape(-1,1)
y_depressive = reg['Depressive'].values.reshape(-1,1)


# In[137]:


# splitting the data into training set and testing set

X_train, X_test, y_depressive_train, y_depressive_test = train_test_split(X, y_depressive, test_size=0.3, random_state=0)


# ### 4. Regression Analysis

# #### a. Anxiety Disorder

# In[138]:


# creating a regression object

regression_anxiety = LinearRegression()


# In[139]:


# fitting the regression object onto the training set

regression_anxiety.fit(X_train, y_anxiety_train)


# In[140]:


# predicting the values of y using X

y_anxiety_predicted = regression_anxiety.predict(X_test)


# In[166]:


# creating a scatterplot that shows the regression line on the test set

plot_test = plt
plot_test.scatter(X_test, y_anxiety_test, color='gray', s = 10)
plot_test.plot(X_test, y_anxiety_predicted, color='red', linewidth =1)
plot_test.title('Social Support vs Prevalence of Anxiety Disorder (on test set)')
plot_test.xlabel('Level of social support')
plot_test.ylabel('Prevalence of anxiety disorder')
plot_test.show()


# In[142]:


# creating objects that contain the model summary statistics

rmse_anxiety = mean_squared_error(y_test, y_anxiety_predicted)
r2_anxiety = r2_score(y_test, y_anxiety_predicted) 


# In[153]:


# printing the model summary statistics to evaluate performance of model

print('Slope:' ,regression_anxiety.coef_)
print('Mean squared error: ', rmse_anxiety)
print('R2 score: ', r2_anxiety)


# In[135]:


# creating a dataframe comparing actual and predicted values of y

data = pd.DataFrame({'Actual': y_anxiety_test.flatten(), 'Predicted': y_anxiety_predicted.flatten()})
data.head(30)


# #### b. Depressive Disorder

# In[144]:


# creating a regression object

regression_depressive = LinearRegression()


# In[146]:


# fitting the regression object onto the training set

regression_depressive.fit(X_train, y_depressive_train)


# In[147]:


# predicting the values of y using X

y_depressive_predicted = regression_depressive.predict(X_test)


# In[156]:


# creating a scatterplot that shows the regression line from the model on the test set

plot_test = plt
plot_test.scatter(X_test, y_depressive_test, color='gray', s = 10)
plot_test.plot(X_test, y_depressive_predicted, color='red', linewidth =1)
plot_test.title('Social Support vs Prevalence of Depressive Disorder (test set)')
plot_test.xlabel('Level of social support')
plot_test.ylabel('Prevalence of depressive disorder')
plot_test.show()


# In[154]:


# creating objects that contain the model summary statistics

rmse_depressive = mean_squared_error(y_test, y_depressive_predicted)
r2_depressive = r2_score(y_test, y_depressive_predicted) 


# In[155]:


# printing the model summary statistics to evaluate performance of model

print('Slope:' ,regression_depressive.coef_)
print('Mean squared error: ', rmse_depressive)
print('R2 score: ', r2_depressive)


# In[158]:


# creating a dataframe comparing actual and predicted values of y

data = pd.DataFrame({'Actual': y_depressive_test.flatten(), 'Predicted': y_depressive_predicted.flatten()})
data.head(30)


# ### 5. Interpretations of the performance of the regression lines

# ANXIETY DISORDER: 
# 
# - Regression line for prevalence of anxiety disorder has a slope of 1.93, mean squared error of 1.90 and an R2 score of 0.069.
# - The positive slope of 1.93 shows a positive relationship between the level of social support and anxiety disorder. 
# - The mean squared error is very high, meaning the distance between the regression line and the data points are large, indicating that the regression line is not an accurate representation of the data. 
# - R2 score is very low, being close to 0. This indicates a poor fit. The model is not a close representation of the data. It is also clear that the relationship is not linear as there is a wider range of prevalence with a higher level of social support.
# 
# DEPRESSIVE DISORDER:
# 
# - Regression line for prevalence of depressive disorder has a slope of -2.84, mean squared error of 2.92 and an R2 score of -0.43.
# - The negative slope of -2.84 shows a negative relationship between the level of social support and depressive disorder. 
# - The mean squared error is again very high, meaning the distance between the regression line and the data points are large, indicating that the regression line is not an accurate representation of the data. 
# - R2 score a negative value, indicating a worse fit than a horizontal line. The model is not a close representation of the data. 
# 
# BOTH: 
# 
# - The limited number of data points (153) could have negatively impacted the model's accuracy. 
# - Both the correlation coefficient (anxiety: 0.4; depressive: -0.4) and the scatterplots showed a lack of strong linear relationship between the respective variables, so a weak regression was expected.

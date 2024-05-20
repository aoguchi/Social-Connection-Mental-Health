#!/usr/bin/env python
# coding: utf-8

# ## 6.6 Sourcing & Analyzing Time-Series Data

# ## Index
# [1. Import Libraries and Datasets](#1.-Import-Libraries-and-Datasets)
# <br>
# [2. Subsetting, wrangling, and cleaning time-series data](#2.-Subsetting,-wrangling,-and-cleaning-time-series-data)
# <br>
# [3. Time series analysis: decomposition](#3.-Time-series-analysis:-decomposition)
# <br>
# [4. Testing for stationarity](#4.-Testing-for-stationarity)
# <br>
# [5. Stationarizing the Data](#5.-Stationarizing-the-Data)

# ### 1. Import Libraries and Datasets

# In[15]:


get_ipython().system('pip install statsmodels')


# In[2]:


get_ipython().system('pip install quandl')


# In[1]:


import quandl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm # Using .api imports the public access version of statsmodels, which is a library that handles 
# statistical models.
import os
import warnings # This is a library that handles warnings.

warnings.filterwarnings("ignore") # Disable deprecation warnings that could indicate, for instance, a suspended library or 
# feature. These are more relevant to developers and very seldom to analysts.

plt.style.use('fivethirtyeight') # This is a styling option for how your plots will appear. More examples here:
# https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html
# https://matplotlib.org/3.1.0/gallery/style_sheets/fivethirtyeight.html


# In[2]:


# This option ensures that the graphs you create are displayed within the notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# import datasets

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath = os.path.join(path, '.csv', 'prevelance.csv')

# assign df name
prev = pd.read_csv(filepath)


# In[4]:


# viewing df prevelance 

prev.head(5)


# In[5]:


prev.shape


# In[6]:


# isolating depression prevelance for only US

prev_dep_us = prev[prev['Entity'] == 'United States']


# In[7]:


prev_dep_us


# In[8]:


# removing irrelevant columns from df 

prev_dep_us = prev_dep_us.drop(columns = ['Unnamed: 2', 'Anxiety', 'Entity'])


# In[9]:


prev_dep_us


# In[10]:


# turning Year column into index 

prev_dep_us.set_index('Year', inplace=True)


# In[11]:


prev_dep_us


# In[12]:


prev_dep_us.info()


# In[13]:


type(prev_dep_us)


# In[14]:


# Plot the data using matplotlib.

plt.figure(figsize=(15,5), dpi=100) # The dpi argument controls the quality of the visualization here. When it's set to 100,
# it will produce lower-than-standard quality, which is useful if, similar to this notebook, you'll have a lot of plots.
# A large number of plots will increase the size of the notebook, which could take more time to load and eat up a lot of RAM!

plt.plot(prev_dep_us)


# ### 2. Subsetting, wrangling, and cleaning time-series data

# In[15]:


# resetting index to use Year column as a filter

prev_dep_us2 = prev_dep_us.reset_index()


# In[16]:


prev_dep_us2.head()


# In[17]:


# checking for missing values

prev_dep_us2.isnull().sum() 


# In[18]:


# checking for duplicates

dups = prev_dep_us2.duplicated()
dups.sum()


# In[19]:


# converting 'year' column to datetime
prev_dep_us2['Year'] = pd.to_datetime(prev_dep_us2['Year'], format='%Y')


# In[20]:


# turning Year column into index 

prev_dep_us2.set_index('Year', inplace=True)


# In[35]:


prev_dep_us2


# In[22]:


# Plot the new data set

plt.figure(figsize=(15,5), dpi=100)
plt.plot(prev_dep_us2)


# ### 3. Time series analysis: decomposition

# In[23]:


# decomposing the time-series using an additive model

decomposition = sm.tsa.seasonal_decompose(prev_dep_us2, model='additive')


# In[24]:


from pylab import rcParams # This will define a fixed size for all special charts.

rcParams['figure.figsize'] = 18, 7


# In[25]:


# Plot the separate components

decomposition.plot()
plt.show()


# Trend: Gradual increase from 1990 to 2000, platueau to 2011, a slight decrease to 2015 and a plateau to 2019
# 
# Seasonality: none because there is no finer granularity like quarterly/monthly data
# 
# Residual: none

# ### 4. Testing for stationarity

# In[26]:


# The adfuller() function will import from the model from statsmodels for the test; however, running it will only return 
# an array of numbers. This is why you need to also define a function that prints the correct output from that array.

from statsmodels.tsa.stattools import adfuller # Import the adfuller() function

def dickey_fuller(timeseries): # Define the function
    # Perform the Dickey-Fuller test:
    print ('Dickey-Fuller Stationarity test:')
    test = adfuller(timeseries, autolag='AIC')
    result = pd.Series(test[0:4], index=['Test Statistic','p-value','Number of Lags Used','Number of Observations Used'])
    for key,value in test[4].items():
       result['Critical Value (%s)'%key] = value
    print (result)

# Apply the test using the function on the time series
dickey_fuller(prev_dep_us2['Depressive'])


# We want to reject the null hypothesis, or disprove the presense of unit root, to prove that the data is stationary, in order to proceed with the forecast. 
# 
# Because the test statistic is larger than the critical value (1, 5, and 10%) the null hypothesis cannot be rejected. This means there is a unit root in the data and the dataset is non-stationary. 

# In[27]:


# another way to test if data is stationary or non-stationary is by testing autocorrelations

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Here, you import the autocorrelation and partial correlation plots

plot_acf(prev_dep_us2)
plt.show()


# "The vertical lines represent the lags in the series, while the blue area represents the confidence interval. When lines go above the blue edge of the confidence interval, this means you have lags that are significantly correlated with each other and when you have many lags beyond this interval, you can deduce that your data is non-stationary. (Depressive) time-series has a couple of lags correlated with each other. This means there’s autocorrelated data and the set is likely non-stationary, which supports the result of the Dickey-Fuller test." ...although there are only three lags outside of the confidence interval?

# ### 5. Stationarizing the Data

# In[28]:


prev_diff = prev_dep_us2 - prev_dep_us2.shift(1) # The df.shift(1) function turns the observation to t-1, making the whole thing t - (t -1)


# In[29]:


prev_diff.dropna(inplace = True) # Here, you remove the missing values that came about as a result of the differencing. 
# You need to remove these or you won't be able to run the Dickey-Fuller test.


# In[30]:


prev_diff.head()


# In[31]:


prev_diff.columns


# In[32]:


# seeing what the differencing did to the time-series curve

plt.figure(figsize=(15,5), dpi=100)
plt.plot(prev_diff)


# In[33]:


dickey_fuller(prev_diff)


# In[34]:


plot_acf(prev_diff)
plt.show()


# Differencing made the test statistic smaller than the 5% and 10% critical value (still larger than 1%) but the autocorrelation looks identical, with the same three lags outside of the confidence interval. 

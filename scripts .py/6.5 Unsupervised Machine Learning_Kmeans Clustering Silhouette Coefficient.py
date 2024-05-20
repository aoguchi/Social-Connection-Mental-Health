#!/usr/bin/env python
# coding: utf-8

# ## 6.5 Unsupervised Machine Learning: K-means Clustering, Silhouette Coefficient

# ## Index
# [1. Import Libraries and Datasets](#1.-Import-Libraries-andDatasets)
# <br>
# [2. Data Cleaning](#2.-Data-Cleaning)
# <br>
# [3. Elbow Technique](#3.-Elbow-Technique)
# <br>
# [4. K-means Clustering](#4.-K-means-Clustering)
# <br>
# [5. K-means Aggregations](#5.-K-means-Aggregations)
# <br>
# [6. Silhouette Coefficient](#6.-Silhouette-Coefficient)

# ### 1. Import Libraries and Datasets

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as cm
import matplotlib.pyplot as plt
import os
import sklearn
import pylab as pl 

from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


# In[3]:


# This option ensures that the graphs you create are displayed within the notebook without the need to "call" them specifically.

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# import datasets

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath = os.path.join(path, '.pkl', 'only_nec.pkl')

# assign df name
df = pd.read_pickle(filepath)


# ### 2. Data Cleaning

# In[5]:


# remove column limit of output 
pd.options.display.max_columns = None

# remove row limit of output 
pd.options.display.max_rows = None


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


# choosing select columns for K-means model; omitting categorical columns
unsup = df[['Anxiety', 'Depressive', 'AHDI', 'VComfortable', 'vs', 'F/F', 'social_support']]


# In[9]:


unsup.head()


# In[10]:


unsup.shape


# In[11]:


# drop all NaN as they are not allowed to run K-means model 
unsup_nonan = unsup.dropna()


# In[12]:


unsup_nonan.shape


# ### 3. Elbow Technique

# In[13]:


# see how many clusters is optimal for this df by first suggesting a range of 1-10
num_cl = range(1, 10)

# define k-means clusters in the above range
kmeans = [KMeans(n_clusters=i) for i in num_cl] 


# In[14]:


# create rate of variation, or score, for each cluster size created above 
score = [kmeans[i].fit(unsup_nonan).score(unsup_nonan) for i in range(len(kmeans))] 

score


# In[15]:


# plot elbow curve using PyLab

pl.plot(num_cl, score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# Number of clusters = 3?? 

# ### 4. K-means Clustering

# In[16]:


# create k-means object w 3 clusters

kmeans = KMeans(n_clusters = 3) 


# In[17]:


# fit k-means object to data

kmeans.fit(unsup_nonan)


# In[18]:


# create column named 'clusters' in unsup_nonan to show cluster group assigned number 

unsup_nonan['clusters'] = kmeans.fit_predict(unsup_nonan)


# In[19]:


unsup_nonan.head()


# In[20]:


unsup_nonan['clusters'].value_counts()


# In[21]:


# plot clusters for 'social_support' and 'anxiety' variables

plt.figure(figsize=(6,4))
ax = sns.scatterplot(x=unsup_nonan['social_support'], y=unsup_nonan['Anxiety'], hue=kmeans.labels_, s=50) 

ax.grid(False)
plt.xlabel('Level of social support')
plt.ylabel('Prevalence of anxiety disorder')
plt.show()


# In[22]:


# plot clusters for 'social_support' and 'depressive' variables

plt.figure(figsize=(6,4))
ax = sns.scatterplot(x=unsup_nonan['social_support'], y=unsup_nonan['Depressive'], hue=kmeans.labels_, s=50) 

ax.grid(False)
plt.xlabel('Level of social support')
plt.ylabel('Prevalence of anxiety disorder')
plt.show()


# In[23]:


# plot clusters for 'VComfortable' and 'depressive' variables

plt.figure(figsize=(6,4))
ax = sns.scatterplot(x=unsup_nonan['VComfortable'], y=unsup_nonan['Depressive'], hue=kmeans.labels_, s=50) 

ax.grid(False)
plt.xlabel('Very comfortable talking about anxiety/depression to friends/family')
plt.ylabel('Prevalence of depressive disorder')
plt.show()


# In[24]:


# plot clusters for 'VComfortable' and 'depressive' variables

plt.figure(figsize=(6,4))
ax = sns.scatterplot(x=unsup_nonan['VComfortable'], y=unsup_nonan['Anxiety'], hue=kmeans.labels_, s=50) 

ax.grid(False)
plt.xlabel('Very comfortable talking about anxiety/depression to friends/family')
plt.ylabel('Prevalence of anxiety disorder')
plt.show()


# ### 5. K-means Aggregations

# In[25]:


unsup_nonan.loc[unsup_nonan['clusters'] == 2, 'clusters'] = '2'
unsup_nonan.loc[unsup_nonan['clusters'] == 1, 'clusters'] = '1'
unsup_nonan.loc[unsup_nonan['clusters'] == 0, 'clusters'] = '0'


# I changed the newly labeled '2 dark purple', '1 purple' and '0 pink' back to numerical values bc I was getting errors when calculating the silouette scores below.
# 
# Should I have removed the 'clusters' column from the unsup_nonan dataframe before calculating the silouette scores below?

# In[26]:


unsup_nonan.groupby('clusters').agg({'Anxiety':['mean', 'median'], 'Depressive':['mean', 'median'], 'AHDI':['mean', 'median'], 'VComfortable':['mean', 'median'], 'vs':['mean', 'median'], 'F/F':['mean', 'median'], 'social_support':['mean', 'median']})


# I initially tried using the df with all 21 variables (with categorical columns removed). The elbow test looked very similar to what it is now, having reduced the variables to 7. 
# 
# Because the elbow test came out more rounded and an optimal number was hard to distinguish, I initially had k=5. Then I reduced it to 3 to see if that would improve the clusters but there were not notable differences.
# 
# The clusters, when plotted on the scatterplot, looks random. The only pattern I saw was when I plotted "very comfortable talking about A/D to F/F" against "anxiety" and "depressive" respectively. 

# ### 6. Silhouette Coefficient

# I looked up the below codes and worked out the errors. I'm not sure if they're correct.

# In[27]:


# Silouette coefficient for each K-mean model

# Range of models to test
cluster_range = [2, 3, 4, 5, 6]

# Create lists to store k-means and silhouette scores
silhouette_scores = []
kmeans_models = []

# Iterate through different cluster numbers
for n_clusters in cluster_range:
    kmeans_sc = KMeans(n_clusters=n_clusters)
    kmeans_sc.fit(unsup_nonan)
    
    # Predict cluster labels for each data point
    cluster_labels = kmeans_sc.predict(unsup_nonan)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(unsup_nonan, cluster_labels)
    
    # Append silhouette score and k-means model to lists
    silhouette_scores.append(silhouette_avg)
    kmeans_models.append(kmeans_sc)
    
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg}")


# In[28]:


# Create a list to store K-mean models 
kmeans_models = []
for num_clusters in range(2, 7):
    kmeans_sc = KMeans(n_clusters=num_clusters)
    kmeans_sc.fit(unsup_nonan)
    kmeans_models.append(kmeans_sc)

# Create a list to store sizes of clusters for each K-mean model
cluster_sizes = []
for kmeans_sc in kmeans_models:
    
    # Predict cluster labels for each data point
    cluster_labels = kmeans_sc.predict(unsup_nonan)
    
    # Count the number of data points in each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes.append(dict(zip(unique, counts)))

# Display the sizes of clusters for each K-means model
for k, sizes in enumerate(cluster_sizes, start=2):
    print(f'Clusters for k={k}: {sizes}')


# The k-mean models with the highest scores are:
# - 2 (0.51)
# - 5 (0.41)
# 
# But looking at the cluster sizes, k=2 has a large difference (75, 21), as does k=5 (42, 24, 13, 9, 8). I'm reading that the best k-mean has both a high score and an even distribution of cluster size. 
# 
# Next up are: 
# - 3 (0.40) with 47, 32, 17
# - 4 (0.36) with 37, 27, 18, 14
# - 6 (0.33) with 36, 18, 15, 10, 9, 8
# 
# ...which all still have imbalanced cluster sizes...

# ### 7. Export Dataframe

# In[29]:


unsup_nonan.to_csv(os.path.join(path, '.csv', 'Tableau kmean cluster.csv'))


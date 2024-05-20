#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing Libraries & Datasets

# In[3]:


# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os


# In[4]:


# checking matplotlib version

matplotlib.__version__


# In[5]:


# import datasets

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath1 = os.path.join(path, '.pkl', 'all_merge.pkl')
filepath2 = os.path.join(path, '.pkl', 'all_merge_numbers.pkl')

# assign df names
ALL = pd.read_pickle(filepath1)
ALL_num = pd.read_pickle(filepath2)


# ### 2. Data Cleaning

# #### a. Changing data types

# In[6]:


# remove column limit of output 

pd.options.display.max_columns = None


# In[7]:


# remove rows limit of output 

pd.options.display.max_rows = None


# In[8]:


ALL_num.head(5)


# In[9]:


# checking data types 

ALL_num.dtypes


# In[10]:


# change some object columns to float64 

ALL_num = ALL_num.astype({'Income_group': 'float64', 'Legis': 'float64', 'Policy': 'float64'})


# In[11]:


ALL_num.dtypes


# Kept getting error messages when I tried to change column 'GDPpc' to float64 or int. It is assigned as object. I exported it as .csv to open and change in Excel and imported it back here but it goes back to object...

# In[12]:


# checking for mix-type data

for col in ALL_num.columns.tolist():
      weird = (ALL_num[[col]].applymap(type) != ALL_num[[col]].iloc[0].apply(type)).any(axis = 1)
      if len (ALL_num[weird]) > 0:
        print (col)


# #### b. Checking for missing values

# In[13]:


ALL_num.isnull().sum()


# #### c. Checking for duplicates

# In[14]:


ALL_num_dups = ALL_num[ALL_num.duplicated()]


# In[15]:


ALL_num_dups


# No duplicates

# #### d. Creating a subset for correlations (removing first four object columns)

# In[16]:


ALL_num_correl = ALL_num.copy()
columns_to_remove = ['Entity', 'Code', 'Continent', 'Region', 'GDPpc', 'Schizophrenia', 'Bipolar', 'Eating', 'Average_ALL', 'Legis', 'Policy']
ALL_num_correl.drop(columns=columns_to_remove, inplace=True)


# In[17]:


ALL_num_correl.head()


# In[18]:


# shortening columns names

ALL_num_correl.rename(columns = {'Anxiety':'anxiety', 'Depressive':'depressive', 'Income_group':'income', '%Invest':'gov_invest', 'F/F_have_been_A/D_YES':'ffAD', 'Very_comfortable_speaking_about_A/D_with_F/F_YES':'comfortable', 'Verycomfortable_vs_Havebeen':'vs', 'talked_to_F/F':'f/f', 'spent_time_in_nature_outdoors':'nature', 'improved_healthy_lifestyle_behaviors':'lifestyle', 'made_change_to_personal_relationships':'relationships', 'made_change_to_work_situation':'work', 'took_prescribed_medication':'meds', 'talked_to_mental_health_professional':'prof','engaged_in_religious_spiritual_activities':'religion', 'Life_ladder':'life_ladder', 'Social_support':'social_support', 'Healthy_life_expectancy_at_birth':'life_expectancy', 'Freedom_to_make_life_choices':'choice_freedom', 'Perceptions_of_corruption':'corruption', 'Positive_affect':'pos', 'Negative_affect':'neg' }, inplace=True)


# In[19]:


ALL_num_correl.head()


# ### 3. Exploring Relationships

# #### a. Correlation

# In[20]:


# create a correlation matrix using pandas

ALL_num_correl.corr()


# In[21]:


# create a correlation heatmap using matplotlib

plt.matshow(ALL_num_correl.corr())
plt.show()


# In[22]:


# save figure
plt.matshow(ALL_num_correl.corr())
plt.savefig("out.png") 

# This will save the image in the working directory. 
# If you don't know what this directory is the next line will show you how to check


# In[23]:


#current dir

cwd = os.getcwd()
cwd


# In[24]:


# add labels, a legend, and change the size of the heatmap

f = plt.figure(figsize=(10, 10)) # figure size 
plt.matshow(ALL_num_correl.corr(), fignum=f.number) # type of plot
plt.xticks(range(ALL_num_correl.shape[1]), ALL_num_correl.columns, fontsize=10, rotation=90) # x axis labels
plt.yticks(range(ALL_num_correl.shape[1]), ALL_num_correl.columns, fontsize=10) # y axis labels
cb = plt.colorbar() # add a colour legend (called colorbar)
cb.ax.tick_params(labelsize=14) # add font size
plt.title('Correlation Matrix', fontsize=12) # add title


# #### Create a correlation heatmap using seaborn

# In[25]:


# create subplot with matplotlib
f,ax = plt.subplots(figsize=(20,20))

# create correlation heatmap in seaborn by applying heatmap onto correlation matrix and subplots defined above
corr = sns.heatmap(ALL_num_correl.corr(), annot = True, fmt=".1f", cmap='coolwarm', vmin=-1.0, vmax=1.0, ax = ax) 

# 'annot' allows plot to place correlation coefficients onto heatmap


# In[26]:


# create subplot with matplotlib
f, ax = plt.subplots(figsize=(20, 20))

# create mask for values outside desired range (-0.3 to 0.3)
mask = (ALL_num_correl.corr() <= -0.3) | (ALL_num_correl.corr() >= 0.3)

# set values outside desired range to NaN
data_masked = ALL_num_correl.corr().where(mask)

# create correlation heatmap in seaborn by applying heatmap onto filtered data
corr = sns.heatmap(data_masked, annot=True, fmt=".1f", cmap='coolwarm', vmin=-1.0, vmax=1.0, ax=ax)

# display plot
plt.show()


# In[27]:


# create subplot with matplotlib
f, ax = plt.subplots(figsize=(20, 20))

# create mask for values outside desired range (-0.3 to 0.3)
mask = (ALL_num_correl.corr() <= -0.3) | (ALL_num_correl.corr() >= 0.3)

# set values outside desired range to NaN
data_masked = ALL_num_correl.corr().where(mask)

# create correlation heatmap in seaborn by applying heatmap onto filtered data
font_settings = {"family": "Calibri", "weight": "normal", "size": 10}
corr = sns.heatmap(data_masked, annot=True, fmt=".1f", cmap='coolwarm', vmin=-1.0, vmax=1.0, ax=ax,
                   annot_kws={"fontdict": font_settings})

# display plot
plt.show()


# In[28]:


# export heatmap as .svg

svg_filename = "coco_heatmap_vector.svg"
full_path = r'C:\Users\pears\Desktop\CF\Achievement 6\visuals\\' + svg_filename
corr.get_figure().savefig(full_path, format="svg")


# In[29]:


# export heatmap as .svg

svg_filename = "coco_heatmap_pdf.svg"
full_path = r'C:\Users\pears\Desktop\CF\Achievement 6\visuals\\' + svg_filename
corr.get_figure().savefig(full_path, format="pdf")


# In[30]:


# create subplot with matplotlib
f, ax = plt.subplots(figsize=(20, 20))

# create mask for values outside desired range (-0.7 to 0.7)
mask = (ALL_num_correl.corr() <= -0.7) | (ALL_num_correl.corr() >= 0.68)

# set values outside desired range to NaN
data_masked = ALL_num_correl.corr().where(mask)

# create correlation heatmap in seaborn by applying heatmap onto filtered data
corr = sns.heatmap(data_masked, annot=True, fmt=".1f", cmap='coolwarm', vmin=-1.0, vmax=1.0, ax=ax)

# display plot
plt.show()


# #### b. Scatterplots

# In[31]:


# create scatterplot in seaborn

sns.lmplot(x = 'Social_support', y = 'Very_comfortable_speaking_about_A/D_with_F/F_YES', data = ALL_num_correl)


# In[ ]:


# create scatterplot in seaborn

sns.lmplot(x = 'AHDI', y = 'Very_comfortable_speaking_about_A/D_with_F/F_YES', data = ALL_num_correl)


# In[ ]:


# create scatterplot in seaborn

sns.lmplot(x = 'AHDI', y = 'F/F_have_been_A/D_YES', data = ALL_num_correl)


# In[ ]:


# create scatterplot for 'depressive' and 'anxiety' columns in seaborn

sns.lmplot(x = 'Social_support', y = 'F/F_have_been_A/D_YES', data = ALL_num_correl)


# In[ ]:


# create scatterplot for 'depressive' and 'anxiety' columns in seaborn

sns.lmplot(x = 'F/F_have_been_A/D_YES', y = 'Social_support', data = ALL_num_correl)


# In[ ]:


# create scatterplot for 'depressive' and 'anxiety' columns in seaborn

sns.lmplot(x = 'Social_support', y = 'Income_group', data = ALL_num_correl)


# #### c. Pair Plots

# In[32]:


# Create a pair plot 

g = sns.pairplot(ALL_num_correl)


# In[35]:


# subset for specific variables 

sub1 = ALL_num_correl[['depressive', 'anxiety', 'social_support', 'life_ladder', 'ffAD', 'comfortable', 'f/f', 'AHDI', 'income', 'gov_invest']]


# In[36]:


# create a pair plot 

g = sns.pairplot(sub1)


# #### d. Categorical Plots

# In[39]:


# define a custom color palette
custom_palette = sns.color_palette(["yellow", "orange", "green", "blue"])

# create categorical plot with the custom color palette
sns.set(style="ticks")
g = sns.catplot(x="social_support", y="comfortable", hue="income", data=ALL_num_correl, palette=custom_palette)

# set x-axis ticks with increments of 10
ax = plt.gca()
ax.set_xticks(range(0, 101, 10))  # for increments of 10

# display plot
plt.show()


# ### 4. Exporting

# In[ ]:


# export ALL_num_correl as pickle:

ALL_num_correl.to_pickle(os.path.join(path, '.pkl', 'all_num_correl.pkl'))


# In[40]:


# export sub1 pair plot as .png:

pair_plot = sns.pairplot(sub1)
pair_plot.savefig(os.path.join(path, 'viz', 'pair_plot_sub1.png'))


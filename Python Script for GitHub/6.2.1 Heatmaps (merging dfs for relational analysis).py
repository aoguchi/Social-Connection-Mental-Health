#!/usr/bin/env python
# coding: utf-8

# ### 1. Import libraries & datasets

# In[3]:


# import libraries
import pandas as pd
import numpy as np
import os


# In[4]:


# import dataset

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath1 = os.path.join(path, '.csv', 'prevelance.csv')
filepath2 = os.path.join(path, '.csv', 'AHDI_GDPpc.csv')
filepath3 = os.path.join(path, '.csv', 'GNIpc_groupings.csv')
filepath4 = os.path.join(path, '.pkl', 'government.pkl')
filepath5 = os.path.join(path, '.csv', 'cultural_attitude.csv')
filepath6 = os.path.join(path, '.csv', 'coping_mechanisms.csv')
filepath7 = os.path.join(path, '.csv', 'happiness_report.csv')

# assign df names
prev = pd.read_csv(filepath1)
AHDIGDP = pd.read_csv(filepath2)
GNI = pd.read_csv(filepath3)
gov = pd.read_pickle(filepath4)
attitude = pd.read_csv(filepath5)
coping = pd.read_csv(filepath6)
happiness = pd.read_csv(filepath7)


# ### 2. Merging df to create one df for relational analysis

# #### a. Creating subsets for most recent year of data collected

# In[19]:


# Prevalence df

prev


# In[9]:


# Prevalence: creating a subset for only 2019 data

prev_sub = prev[prev['Year']==2019]


# In[7]:


prev_sub


# 6,150 rows reduced to 205 

# In[18]:


prev_sub.count()


# In[5]:


# AHDI & GDPpc

AHDIGDP


# In[10]:


# AHDI: creating a subset for only 2020 data

AHDI_sub = AHDIGDP[AHDIGDP['Year']==2020]


# In[12]:


AHDI_sub


# In[17]:


AHDI_sub.count()


# In[22]:


# GDPpc: creating a subset for only 2018 data

GDPpc_sub = AHDIGDP[AHDIGDP['Year']==2018]


# In[23]:


GDPpc_sub


# In[24]:


GDPpc_sub.count()


# In[31]:


# happiness index

happiness


# In[35]:


# happiness index: creating a subset for only 2017 data

happiness_sub = happiness[happiness['Year']==2017]


# In[36]:


happiness_sub


# In[37]:


happiness_sub['Entity'].nunique()


# #### b. Merging dfs

# In[48]:


GDPpc_sub


# In[58]:


# merge GDPpc_sub onto prev_sub using Entity has key; add merge flag

prev_sub_merge = prev_sub.merge(GDPpc_sub[['Entity','GDPpc']], on = 'Entity', how = 'left', indicator=True)


# In[59]:


prev_sub_merge


# In[64]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[60]:


AHDI_sub


# In[66]:


# merge AHDI onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(AHDI_sub[['Entity','AHDI']], on = 'Entity', how = 'left', indicator=True)


# In[69]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[70]:


prev_sub_merge


# In[72]:


# merge GNI onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(GNI[['Entity','Income_group']], on = 'Entity', how = 'left', indicator=True)


# In[76]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[78]:


prev_sub_merge


# In[82]:


gov.head(5)


# In[83]:


# merge gov onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(gov[['Entity','Legis', 'Policy', '%Invest']], on = 'Entity', how = 'left', indicator=True)


# In[84]:


prev_sub_merge.head(10)


# In[87]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[90]:


attitude.head(10)


# In[91]:


# merge attitude onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(attitude[['Entity','F/F_have_been_A/D_YES', 'Very_comfortable_speaking_about_A/D_with_F/F_YES', 'Verycomfortable_vs_Havebeen']], on = 'Entity', how = 'left', indicator=True)


# In[93]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[94]:


prev_sub_merge


# In[95]:


coping


# In[96]:


# merge coping onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(coping[['Entity','talked_to_F/F', 'spent_time_in_nature_outdoors', 'improved_healthy_lifestyle_behaviors', 'made_change_to_personal_relationships', 'made_change_to_work_situation', 'took_prescribed_medication', 'talked_to_mental_health_professional', 'engaged_in_religious_spiritual_activities']], on = 'Entity', how = 'left', indicator=True)


# In[98]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[100]:


happiness_sub


# In[101]:


# merge happiness_sub onto prev_sub_merge using Entity has key; add merge flag

prev_sub_merge = prev_sub_merge.merge(happiness_sub[['Entity', 'Life_ladder', 'Social_support', 'Healthy_life_expectancy_at_birth', 'Freedom_to_make_life_choices', 'Perceptions_of_corruption', 'Positive_affect', 'Negative_affect']], on = 'Entity', how = 'left', indicator=True)


# In[102]:


prev_sub_merge


# In[103]:


# deleting '_merge' column 

prev_sub_merge.drop(columns = ['_merge'], inplace=True)


# In[144]:


# deleting 'Year' column

prev_sub_merge.drop(columns = ['Year'], inplace=True)


# In[145]:


prev_sub_merge


# ### 3. Changing Yes to 1, No to 0 (for Legis and Policy columns)

# In[106]:


# remove column limit of output 

pd.options.display.max_columns = None


# In[107]:


prev_sub_merge.head(5)


# In[142]:


# replacing yes with 1, no with 1 for columns Legis and Policy; replacing GNI income_groups with 1~4

prev_sub_merge_num = prev_sub_merge_num = prev_sub_merge.replace({'Legis': {'Yes': '1', 'No': '0'}, 'Policy': {'Yes': '1', 'No': '0'}, 'Income_group': {'Low income': '1', 'Lower middle income': '2', 'Upper middle income': '3', 'High income': '4'}})


# In[140]:


prev_sub_merge_num.head(10)


# In[147]:


# deleting 'Year' column

prev_sub_merge_num.drop(columns = ['Year'], inplace=True)


# In[148]:


prev_sub_merge_num


# ### 4. Exporting

# In[146]:


# export prev_sub_merge as pickle

prev_sub_merge.to_pickle(os.path.join(path, '.pkl', 'all_merge.pkl'))


# In[149]:


# export prev_sub_merge_num as pickle

prev_sub_merge_num.to_pickle(os.path.join(path, '.pkl', 'all_merge_numbers.pkl'))


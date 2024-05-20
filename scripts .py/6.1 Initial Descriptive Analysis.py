#!/usr/bin/env python
# coding: utf-8

# ### import libraries & data

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import os


# In[121]:


# import dataset

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath1 = os.path.join(path, '.csv', 'AHDI_GDPpc.csv')
filepath2 = os.path.join(path, '.csv', 'continent_region_codes.csv')
filepath3 = os.path.join(path, '.csv', 'coping_mechanisms.csv')
filepath4 = os.path.join(path, '.csv', 'cultural_attitude.csv')
filepath5 = os.path.join(path, '.csv', 'GNIpc_groupings.csv')
filepath6 = os.path.join(path, '.csv', 'gov_investment.csv')
filepath7 = os.path.join(path, '.csv', 'happiness_report.csv')
filepath8 = os.path.join(path, '.csv', 'legislation_status.csv')
filepath9 = os.path.join(path, '.csv', 'policy_status.csv')
filepath10 = os.path.join(path, '.csv', 'prevelance.csv')

# assign df names
HDIGDP = pd.read_csv(filepath1)
regcodes = pd.read_csv(filepath2)
coping = pd.read_csv(filepath3)
attitude = pd.read_csv(filepath4)
GNI = pd.read_csv(filepath5)
invest = pd.read_csv(filepath6)
happiness = pd.read_csv(filepath7)
legis = pd.read_csv(filepath8)
policy = pd.read_csv(filepath9)
prev = pd.read_csv(filepath10)


# ### 1. AHDI & GDPpc

# In[54]:


HDIGDP.head(5)


# In[55]:


# checking data type

HDIGDP.info()


# In[96]:


# checking if all countries have same quantity of rows

HDIGDP['Entity'].value_counts()


# All countries have 30 entries, ones with 29 are missing 2019 data, ones with 7 only have AHDI data and no GDPpc.

# In[178]:


# checking count of each column 

HDIGDP.count()


# In[181]:


# counting how many unique entities are in this df

HDIGDP['Entity'].nunique()


# ### 3. coping mechanisms

# In[52]:


coping.head(5)


# In[57]:


# checking data type

coping.info()


# In[71]:


# descriptive analysis

coping.describe()


# In[100]:


# checking if all countries have same quantity of rows 

coping['Entity'].value_counts()


# In[83]:


# checking count of each column

coping.count()


# In[135]:


# aggregating for 'talked to friends and families' as a coping mechanism; the highest result

coping.groupby('Region').agg({'talked_to_F/F': ['mean','min','max']})


# In[182]:


# counting how many unique entities are in this df

coping['Entity'].nunique()


# ### 4. cultural attitude

# In[63]:


attitude.head(5)


# In[64]:


# checking data types

attitude.info()


# In[65]:


# descriptive analysis 

attitude.describe()


# In[102]:


# checking count of each column

attitude.count()


# In[183]:


# counting how many unique entities are in this df

attitude['Entity'].nunique()


# ### 5. GNIpc

# In[84]:


GNI.head(5)


# In[85]:


GNI.info()


# In[107]:


GNI.count()


# In[118]:


# checking count of each 'income_group'

GNI['Income_group'].value_counts().sort_index()


# In[116]:


# group by combinations of 'income group' and 'continent'

GNI.groupby(['Income_group', 'Continent']).size()


# In[184]:


# counting how many unique entities are in this df

GNI['Entity'].nunique()


# ### 6. percentage government investment in mental health care, out of a larger health care budget

# In[119]:


invest.head(5)


# In[120]:


invest.info()


# In[121]:


invest.describe()


# An average of 3.4% of the larger healthcare budget is being invested into mental healthcare. An average prevalence of the five mental disorder diagnoses is 1.9%. Prevalence does not include those who experience mental struggles who are not officially diagnosed. Would like to see correlation between prevalence and government investment.

# In[176]:


invest.count()


# In[113]:


# counting how many unique entities are in this df

invest['Entity'].nunique()


# ### 7. happiness index 

# In[122]:


happiness.head(5)


# In[136]:


happiness.info()


# In[137]:


happiness.describe()


# In[138]:


happiness.count()


# In[190]:


# counting how many unique entities are in this df

happiness['Entity'].nunique()


# ### 8. legislation 

# In[62]:


legis.head(5)


# In[63]:


legis.info()


# In[64]:


legis.count()


# In[65]:


# counting how many countries have/don't have mental health legislation

legis['MH_Legislation'].value_counts()


# In[66]:


# counting how many countries in each region

legis['Region'].value_counts()


# In[67]:


# how many YES and NO in each region?

legis.groupby(['MH_Legislation','Region']).size()


# In[68]:


# counting how many unique entities are in this df

legis['Entity'].nunique()


# ### 9. policy

# In[168]:


policy.head()


# In[169]:


policy.info()


# In[170]:


policy.count()


# In[171]:


# counting how many countries have/don't have mental health policy

policy['MH_Policy'].value_counts()


# In[174]:


# how many YES and NO in each region?

policy.groupby(['MH_Policy','Region']).size()


# In[81]:


# counting how many unique entities are in this df

policy['Entity'].nunique()


# ### 10. prevalence

# In[53]:


prev.head(5)


# In[28]:


prev.info()


# In[29]:


prev.describe()


# In[175]:


prev.count()


# In[189]:


# counting how many unique entities are in this df

prev['Entity'].nunique()


# ### merge: government investment + legislation status + policy status

# In[82]:


invest.head()


# In[83]:


legis.head()


# In[84]:


policy.head()


# In[85]:


# merge policy onto legis using Entity has key; add merge flag

legis_policy = legis.merge(policy, on = ['Entity'], how = 'left', indicator = True)


# In[86]:


legis_policy.head()


# In[87]:


legis_policy['_merge'].value_counts()


# In[88]:


df =  legis_policy[legis_policy['_merge']=='left_only']


# In[89]:


df


# In[90]:


# subset of only necessary columns

legis_policy_subset = legis_policy[['Entity','Code_x','Continent_x', 'Region_x', 'MH_Legislation', 'MH_Policy']]


# In[91]:


legis_policy_subset


# In[98]:


# rename column names

legis_policy_subset.rename(columns = {'Code_x' : 'Code', 'Continent_x' : 'Continent', 'Region_x' : 'Region', 'MH_Legislation':'Legis', 'MH_Policy':'Policy'}, inplace=True)


# In[99]:


legis_policy_subset


# In[100]:


# merge investment onto legis_policy_subset using Entity has key; add merge flag

legis_policy_invest = legis_policy_subset.merge(invest, on = ['Entity'], how = 'left', indicator = True)


# In[101]:


legis_policy_invest


# In[102]:


legis_policy_invest['_merge'].value_counts()


# In[105]:


# subset of only necessary columns

legis_policy_invest_subset = legis_policy_invest[['Entity', 'Code_x', 'Continent_x', 'Region_x', 'Legis', 'Policy', '%']]


# In[106]:


legis_policy_invest_subset


# In[109]:


# rename column names

legis_policy_invest_subset.rename(columns = {'Code_x':'Code', 'Continent_x':'Continent', 'Region_x':'Region', '%':'%Invest'}, inplace=True)


# In[125]:


legis_policy_invest_subset


# ### export: legis_policy_invest_subset

# In[123]:


# export legis_policy_invest_subset as pickle

legis_policy_invest_subset.to_pickle(os.path.join(path, '.pkl', 'government.pkl'))


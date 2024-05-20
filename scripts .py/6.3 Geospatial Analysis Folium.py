#!/usr/bin/env python
# coding: utf-8

# ### 1. Importing libraries & datasets

# In[1]:


# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import folium
import json


# In[2]:


# import datasets

# create path for dataset 
path = r'C:\Users\pears\Desktop\CF\Achievement 6\datasets'

# create filepaths
filepath1 = os.path.join(path, '.pkl', 'all_merge_numbers2.pkl')
filepath2 = os.path.join(path, 'JSON', 'world-countries.json')
# geoJSON file from https://www.kaggle.com/datasets/ktochylin/world-countries/

# assign df names
df = pd.read_pickle(filepath1)
geo = pd.read_json(filepath2)


# In[3]:


# view geo json

f = open(r'C:\Users\pears\Desktop\CF\Achievement 6\datasets\JSON\world-countries.json',)
  
# returns JSON object as dictionary
geo = json.load(f)

# iterating through json list
for i in geo['features']:
    print(i)


# ### 2. Wrangling Data

# In[4]:


# remove column limit of output 
pd.options.display.max_columns = None

# remove row limit of output 
pd.options.display.max_rows = None


# In[5]:


df.head(5)


# In[6]:


# choosing select columns for choropleth map

choro = df[['Entity','Continent','Anxiety','Depressive', 'AHDI', 'Social_support', 'F/F_have_been_A/D_YES', 'Very_comfortable_speaking_about_A/D_with_F/F_YES', 'Verycomfortable_vs_Havebeen', 'talked_to_F/F', 'spent_time_in_nature_outdoors', 'improved_healthy_lifestyle_behaviors', 'made_change_to_personal_relationships', 'made_change_to_work_situation', 'took_prescribed_medication', 'talked_to_mental_health_professional', 'engaged_in_religious_spiritual_activities', 'Positive_affect', 'Negative_affect']]


# In[7]:


# shortening columns names

choro.rename(columns = {'Entity':'entity','Continent':'continent','Anxiety':'anxiety', 'Depressive':'depressive', 'Social_support':'social', 'F/F_have_been_A/D_YES':'ffAD', 'Very_comfortable_speaking_about_A/D_with_F/F_YES':'comfortable', 'Verycomfortable_vs_Havebeen':'vs', 'talked_to_F/F':'f/f', 'spent_time_in_nature_outdoors':'nature', 'improved_healthy_lifestyle_behaviors':'lifestyle', 'made_change_to_personal_relationships':'relationship', 'made_change_to_work_situation':'work', 'took_prescribed_medication':'meds', 'talked_to_mental_health_professional':'prof','engaged_in_religious_spiritual_activities':'religion', 'Positive_affect':'pos', 'Negative_affect':'neg' }, inplace=True)


# In[8]:


choro.head(5)


# In[9]:


choro.dtypes


# In[10]:


# dropping duplicate Congo row

choro.drop_duplicates(inplace=True)


# In[11]:


# changing entity names to match geoJson

choro.replace({'entity': {'United States': 'United States of America', 'Congo': 'Republic of the Congo', 'Democratic Republic of Congo':'Democratic Republic of the Congo', 'Tanzania':'United Republic of Tanzania'}}, inplace=True)


# In[12]:


choro


# ### 3. Conduct consistency checks

# In[13]:


# Check for missing values

choro.isnull().sum()


# In[14]:


# descriptive analysis for anxiety prevalence 

choro.describe()


# In[15]:


# histogram of anxiety disorder prevalence 

sns.histplot(choro['anxiety'], bins=20, kde = True)
plt.xlim(1, 10)


# In[16]:


# histogram of depressive disorder prevalence 

sns.histplot(choro['depressive'], bins=15, kde = True)
plt.xlim(1, 10)


# In[17]:


# histogram of social support

sns.histplot(choro['social'], bins=20, kde = True)
plt.xlim(0, 1)


# In[18]:


# histogram of % of those very comfortable talking to f/f about A/D themselves

sns.histplot(choro['ffAD'], bins=20, kde = True)
plt.xlim(1, 100)


# In[19]:


# histogram of % of those very comfortable talking to f/f about A/D themselves

sns.histplot(choro['comfortable'], bins=20, kde = True)
plt.xlim(1, 100)


# In[20]:


# histogram of those who know others w A/D vs very comfortable talking to f/f about A/D themselves

sns.histplot(choro['vs'], bins=20, kde = True)


# In[21]:


# histogram of % of those who coped by talking to friends/family 

sns.histplot(choro['f/f'], bins=20, kde = True)
plt.xlim(1, 100)


# In[22]:


# histogram of % of those who coped by spending time outdoors/in nature 

sns.histplot(choro['nature'], bins=20, kde = True)
plt.xlim(1, 100)


# In[23]:


# histogram of % of those who coped by improving healthy lifestyle behaviors

sns.histplot(choro['lifestyle'], bins=20, kde = True)
plt.xlim(1, 100)


# In[24]:


# histogram of % of those who coped by making changes in personal relationships

sns.histplot(choro['relationship'], bins=20, kde = True)
plt.xlim(1, 100)


# In[25]:


# histogram of % of those who coped by making changes in work situations

sns.histplot(choro['work'], bins=20, kde = True)
plt.xlim(1, 100)


# In[26]:


# histogram of % of those who coped by taking prescription meds

sns.histplot(choro['meds'], bins=20, kde = True)
plt.xlim(1, 100)


# In[27]:


# histogram of % of those who coped by seeing medical professional

sns.histplot(choro['prof'], bins=20, kde = True)
plt.xlim(1, 100)


# In[28]:


# histogram of % of those who coped by seeing medical professional

sns.histplot(choro['religion'], bins=20, kde = True)
plt.xlim(1, 100)


# ### 4. Plotting choropleth maps

# #### a. Prevalence of anxiety disorder

# In[29]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'anxiety'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "prevalence of anxiety disorder"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### b. Prevalence of depressive disorder

# In[30]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'depressive'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "prevalence of depressive disorder"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### c. AHDI

# In[31]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'AHDI'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "AHDI"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### d. Rating of social support

# In[32]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'social'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "rating of social support"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### e. Those who are "very comfortable" talking to friends/family about their anxiety/depression

# In[33]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'comfortable'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those comfortable talking to friends/family about anxiety/depression"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### f. Those who are "very comfortable" talking to friends/family about their anxiety/depression vs those who know other friends/family with anxiety/depression

# In[34]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'vs'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those comfortable talking to F/F about A/D vs know other F/F with A/D"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# Majority of the world has F/F with A/D, yet do not feel very comfortable talking to F/F about their own A/D. Laos is the only country with a much higher rate. Also scored high for "very comfortable" talking to F/F about A/D.

# #### g. Those who coped with anxiety/depression by talking to friends/family

# In[35]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'f/f'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by talking to friends/family"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# The % of "very comfortable" does not match the % of those who coped by talking to F/F (correlation coefficient of -0.03). This may show that even if people are not "very comfortable", it is still the most accessible form of coping and the most used coping mechanism. People don't need to feel "very comfortable" to reach out to F/F when they need to.

# #### h. Those who coped with anxiety/depression by spending time outside/in nature

# In[36]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'nature'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by spending time outside/in nature"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### i. Those who coped with anxiety/depression by improving healthy lifestyle behaviors  

# In[37]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'lifestyle'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by improving healthy lifestyle behaviors"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### j. Those who coped with anxiety/depression by making changes to personal relationships 

# In[38]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'relationship'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by making changes to personal relationships"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### k. Those who coped with anxiety/depression by making changes to work situations

# In[39]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'work'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by making changes to work situations"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### l. Those who coped with anxiety/depression by taking prescription medications

# In[40]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'meds'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by taking prescription medications"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### m. Those who coped with anxiety/depression by seeing a medical professional

# In[41]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'prof'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by seeing a medical professional"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### n. Those who coped with anxiety/depression by engagining in religious/spiritual activities

# In[42]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'religion'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who coped with anxiety/depression by engagining in religious/spiritual activities"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### o. % of those who rate their previous day as being positive

# In[43]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'pos'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "% of those who rate their previous day as being overall positive"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# #### p. % of those who rate their previous day as being negative

# In[44]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'neg'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "#### o. % of those who rate their previous day as being overall negative"
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# ### 5. Exporting choro df

# In[45]:


choro.to_pickle(os.path.join(path, '.pkl', 'all_choro.pkl'))


# ### MULTI LAYERS IN SINGLE CHOROPLETH

# In[46]:


# setup folium map at high-level zoom
map = folium.Map(location = [50, 0], zoom_start = 1)

# choropleth maps bind Pandas dfs and json geometries for ANXIETY
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'anxiety'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "prevalence of anxiety disorder", 
    name='anxiety'
    ).add_to(map)

# choropleth maps bind Pandas dfs and json geometries for DEPRESSIVE
folium.Choropleth(
    geo_data = geo, 
    data = choro,
    columns = ['entity', 'depressive'],
    key_on = 'feature.properties.name',
    fill_color = 'RdYlGn_r', 
    nan_fill_color = 'transparent', 
    fill_opacity=0.5, 
    line_opacity=0.1,
    legend_name = "prevalence of anxiety disorder",
    name='depressive'
    ).add_to(map)
folium.LayerControl().add_to(map) 

map


# In[ ]:





# In[ ]:





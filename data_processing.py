#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import os
import math
import datetime
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from itertools import groupby
from geopy.distance import distance
from collections import Counter

from predictability import *

pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


data_dir = "/data/users_data/douglas/data/"
df_name = "shanghai"
df_path = os.path.join(data_dir, df_name + "_pandas.tsv") 
df_type = "next_place"


# # Data preprocessing

# In[3]:


def load_and_filter_data(dataset_name):
    """
    Read the data and create a dataframe.
    """
    df = pd.DataFrame(pd.read_csv(dataset_name, encoding='utf-8', sep="\t"))

    # Filter groups according to size.
    df = df.groupby('device_id').filter(lambda x: len(x) >= 170 and len(x) < 6000 and len(set(x)) >= 2) 
    df.reset_index(drop=True, inplace=True)

    # Remove rows with NAs.
    df.dropna(inplace=True)

    # Convert column to datetime format.
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors="raise")
    
    return df


def create_grid(df, grid_side_length):
    """
    Assigns a grid identifier to every location in the dataset.
    The cells of the grid are squares whose side has length 'grid_side_length'.
    """
    bottom_left = (min(df['lat']), min(df['lon']))
    bottom_right = (min(df['lat']), max(df['lon']))
    top_left = (max(df['lat']), min(df['lon']))
    top_right = (max(df['lat']), max(df['lon']))

    nr_rows = int(distance(bottom_left, top_left).meters // grid_side_length)
    nr_cols = int(distance(bottom_left, bottom_right).meters // grid_side_length)

    rows = np.linspace(bottom_left[0], top_left[0], num = nr_rows)
    cols = np.linspace(bottom_left[1], bottom_right[1], num = nr_cols)
    
    df['row'] = np.searchsorted(cols, df['lon'])
    df['col'] = np.searchsorted(rows, df['lat'])
    
    df['grid_id' + str(grid_side_length)] = df['row'] * nr_rows + df['col']
    df['grid_id' + str(grid_side_length)] = pd.to_numeric(df['grid_id' + str(grid_side_length)], downcast='integer')
    df['grid_id' + str(grid_side_length)] = df['grid_id' + str(grid_side_length)].apply(str)
    
    return df


# In[4]:


# Preprocess the data.
df = load_and_filter_data(df_path)

# Create grid of default size (200m side length)
df = create_grid(df, 200)

# Create grids with other sizes. We only do this for the GPS dataset.
if df_name == "macaco":
    for i in range(300, 1100, 100):
        print("Creating grid of side length = {} meters".format(i))
        df = create_grid(df, i)


# In[5]:


# Drop unnecessary columns and rename default grid column.
df.rename(columns={'grid_id200':'grid_id'}, inplace=True)
df.drop(['lat', 'lon', 'row', 'col'], axis=1, inplace=True)


# In[6]:


df.head()


# In[7]:


# Filter the dataset in case the prediction task is next place prediction.
if df_type == "next_place":
    df = df.loc[(df['grid_id'].shift() != df['grid_id'])]


# In[8]:


# Prints basic information about the dataset.
print("There are {} users in the dataset.".format(len(set(df['device_id']))))

avg_seq_size = np.mean([len(grp['grid_id']) for _, grp in df.groupby('device_id')])
stddev_seq_size = np.std([len(grp['grid_id']) for _, grp in df.groupby('device_id')])
print("Average sequence size: {}, and standard deviation: {}".format(avg_seq_size, stddev_seq_size))


# ## Compute Regularity, Stationarity and Diversity

# In[14]:


def compute_metrics(df, colname):
    """
    Compute some mobility-related metrics: stationarity, regularity, and diversity of trajectories.
    """
    regularities = defaultdict()
    stationarities = defaultdict()
    diversities = defaultdict()

    nr_users = len(set(df['device_id']))
    curr_user = 0
    for device_id, grp in df.groupby(df['device_id']):
        curr_user += 1        
        if curr_user % 20 == 0:
            print("Processing user {}/{}".format(str(curr_user), str(nr_users)))
        
        locs = [str(row[colname]) for _, row in grp.iterrows()]

        # Compute metrics.
        regularities[device_id] = regularity(locs)
        stationarities[device_id] = stationarity(locs)
        diversities[device_id] = diversity(locs)
    return regularities, stationarities, diversities


# ## Compute metrics for the default grid length (200 meters)

# In[10]:


regularities, stationarities, diversities = compute_metrics(df, 'grid_id')


# In[11]:


df['regularity'] = df.apply(lambda row: regularities[row['device_id']], axis=1)
df['stationarity'] = df.apply(lambda row: stationarities[row['device_id']], axis=1)
df['diversity'] = df.apply(lambda row: diversities[row['device_id']], axis=1)


# In[12]:


df.head()


# ## Compute metrics for other grid lengths (300 meters to 1 km)

# In[13]:


if df_name == "macaco":
    for i in range(300, 1100, 100):
        print("Computing metrics for grid of side length = {} meters".format(i))
        s = str(i)
        regularities, stationarities, diversities = compute_metrics(df, 'grid_id' + s)
        df['regularity' + s] = df.apply(lambda row: regularities[row['device_id']], axis=1)
        df['stationarity' + s] = df.apply(lambda row: stationarities[row['device_id']], axis=1)
        df['diversity' + s] = df.apply(lambda row: diversities[row['device_id']], axis=1)
    


# ## Compute Entropy and Predictability

# In[19]:


def compute_entropy_and_predictability(df, colname):
    base_entropy = defaultdict()
    actual_entropy = defaultdict()
    baseline_predictability = defaultdict()
    actual_predictability = defaultdict()
    
    nr_users = len(set(df['device_id']))
    curr_user = 0
    for device_id, grp in df.groupby(df['device_id']):
        curr_user += 1        
        if curr_user % 20 == 0:
            print("Processing user {}/{}".format(str(curr_user), str(nr_users)))
        
        # Filter and convert the list of locations to a list of strings, which 
        # is the necessary format for the functions that compute the entropy.
        locations = list(grp[colname])
        sequence = [str(location) for location in locations]

        # Adjust the size of the sequence so that we don't interfere with
        # the calculations of the metrics that depend on the size.
        n = len(set(sequence))

        base_entropy[device_id] = baseline_entropy(sequence)
        actual_entropy[device_id] = entropy_kontoyiannis(sequence)
        actual_predictability[device_id] = max_predictability(actual_entropy[device_id], n)
        baseline_predictability[device_id] = max_predictability(base_entropy[device_id], n)

    return base_entropy, actual_entropy, baseline_predictability, actual_predictability


# In[21]:


base_entropy, actual_entropy, baseline_predictability, actual_predictability = compute_entropy_and_predictability(df, 'grid_id')


# In[22]:


df['baseline_entropy'] = df.apply(lambda row: base_entropy[row['device_id']], axis=1)
df['actual_entropy'] = df.apply(lambda row: actual_entropy[row['device_id']], axis=1)
df['baseline_predictability'] = df.apply(lambda row: baseline_predictability[row['device_id']], axis=1)
df['actual_predictability'] = df.apply(lambda row: actual_predictability[row['device_id']], axis=1)


# In[23]:


if df_name == "macaco":
    for i in range(300, 1100, 100):
        print("Computing entropy for grid of side length = {} meters".format(i))
        s = str(i)
        base_entropy, actual_entropy, baseline_predictability, actual_predictability = compute_entropy_and_predictability(df, 'grid_id' + s)
        df['baseline_entropy' + s] = df.apply(lambda row: base_entropy[row['device_id']], axis=1)
        df['actual_entropy' + s] = df.apply(lambda row: actual_entropy[row['device_id']], axis=1)
        df['baseline_predictability' + s] = df.apply(lambda row: baseline_predictability[row['device_id']], axis=1)
        df['actual_predictability' + s] = df.apply(lambda row: actual_predictability[row['device_id']], axis=1)    


# ## Context

# In[28]:


def compute_entropy_and_predictability_with_context(df):
    stats = defaultdict(list)
    nr_users = len(set(df['device_id']))
    curr_user = 0
    for device_id, grp in df.groupby(df['device_id']):
        curr_user += 1        
        if curr_user % 20 == 0:
            print("Processing user {}/{}".format(str(curr_user), str(nr_users)))
        
        # Build the sequences for which we will compute entropy and predictability
        sequence = [str(item) for item in grp['grid_id']]
        weekday_contexts = [str(item.weekday()) for item in grp['timestamp']]
        hourofday_contexts = [str(item.hour) for item in grp['timestamp']]
        
        # Sequence splitting
        entropy_seq_split_weekday = sequence_splitting(sequence, weekday_contexts)
        entropy_seq_split_hourofday = sequence_splitting(sequence, hourofday_contexts)
        predictability_seq_split_weekday = max_predictability(entropy_seq_split_weekday, len(set(sequence)))
        predictability_seq_split_hourofday = max_predictability(entropy_seq_split_hourofday, len(set(sequence)))
        
        # Sequence merging
        entropy_seq_merge_weekday = sequence_merging(sequence, weekday_contexts)
        entropy_seq_merge_hourofday = sequence_merging(sequence, hourofday_contexts)
        predictability_seq_merge_weekday = max_predictability(entropy_seq_merge_weekday, len(set(sequence)))
        predictability_seq_merge_hourofday = max_predictability(entropy_seq_merge_hourofday, len(set(sequence)))
        
        # Add metrics to dictionary to be processed later
        stats[device_id].append((entropy_seq_split_weekday, entropy_seq_merge_weekday, predictability_seq_split_weekday, predictability_seq_merge_weekday))
        stats[device_id].append((entropy_seq_split_hourofday, entropy_seq_merge_hourofday, predictability_seq_split_hourofday, predictability_seq_merge_hourofday))
        
        # This code will execute only for CDR dataset, which has weather information
        if 'weather_main' in df.columns:
            weather_contexts = [str(item) for item in grp['weather_main']]
            
            # Sequence splitting for weather info
            entropy_seq_split_weather = sequence_splitting(sequence, weather_contexts)
            predictability_seq_split_weather = max_predictability(entropy_seq_split_weather, len(set(sequence)))

            # Sequence merging for weather info            
            entropy_seq_merge_weather = sequence_merging(sequence, weather_contexts)
            predictability_seq_merge_weather = max_predictability(entropy_seq_merge_weather, len(set(sequence)))
            
            stats[device_id].append((entropy_seq_split_weather, entropy_seq_merge_weather, predictability_seq_split_weather, predictability_seq_merge_weather))
        # For the GPS dataset, we fill in weather values with zeroes, so that they will be ignored later
        else:
            stats[device_id].append((0, 0, 0, 0))

    return stats


# In[29]:


stats = compute_entropy_and_predictability_with_context(df)


# In[ ]:


context_types = ['weekday', 'hourofday']
if 'weather_main' in df.columns: # Will execute only for the CDR dataset
    context_types.append('weather')

# Compute context-related metrics (entropy and predictability) for the sequences of contexts
for user_i, context in enumerate(context_types):
    df['entropy_seq_split_' + context] = df.apply(lambda row: stats[row['device_id']][user_i][0], axis=1)
    df['entropy_seq_merge_' + context] = df.apply(lambda row: stats[row['device_id']][user_i][1], axis=1)
    df['predictability_seq_split_' + context] = df.apply(lambda row: stats[row['device_id']][user_i][2], axis=1)
    df['predictability_seq_merge_' + context] = df.apply(lambda row: stats[row['device_id']][user_i][3], axis=1)    


# In[ ]:


df.head()


# In[ ]:


# Write the final dataset to be analyzed by a separate script.
df.to_csv(os.path.join(data_dir, df_name + '_tsas_' + df_type + '.tsv'), sep='\t', encoding='utf-8', index=False)


# In[ ]:





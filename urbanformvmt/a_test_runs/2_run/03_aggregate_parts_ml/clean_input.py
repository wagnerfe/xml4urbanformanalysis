"""
cleans the data according to the four test cases 
1. only 500m<tripdistancemeter<100km
2. only weekday trips
3. only non commercial trips
4. only 500m<tripdistancemeter<100km & only weekday trips & only non commercial trips

@authors: Felix W
last modified: 22.03.2021

"""

## Imports
import pandas as pd
import numpy as np
import os
import sys

# Define paths
path_to_data = '/p/projects/vwproject/felix_files/data/output/ml/model_streets_raw.csv'
path_out = '/p/projects/vwproject/felix_files/data/output/ml/'
df = pd.read_csv(path_to_data)

## Data Cleaning
# (0) Discard useless columns
# Discard useless columns
df.drop(['buffer_part','part','has_buffer'], axis=1, inplace=True)

# (1) Set lower and upper bounds
lmin = 500
lmax = 100000
df_cleaned_lb_ub = df[df['tripdistancemeters'].between(lmin, lmax)]
df_cleaned_lb_ub = df_cleaned_lb_ub.reset_index(drop=True)
# check 
print('Number of trips after applying bounds:',len(df_cleaned_lb_ub))
# take 250k sample
df_cleaned_lb_ub_sample = df_cleaned_lb_ub.sample(n=250000)
df_cleaned_lb_ub_sample = df_cleaned_lb_ub_sample.reset_index(drop=True)
# save to disk
df_cleaned_lb_ub_sample.to_csv(os.path.join(path_out,'model_in_lb_ub.csv'), index=False)


# (2) only weekday
df_cleaned_week = df[df['startdate'].str.match('weekday')]
df_cleaned_week = df_cleaned_week.reset_index(drop=True)
# check
print('Number of trips after filter weekdays:',len(df_cleaned_week))
# take 250k sample
df_cleaned_week_sample = df_cleaned_week.sample(n=250000)
df_cleaned_week_sample = df_cleaned_week_sample.reset_index(drop=True)
# save to disk
df_cleaned_week_sample.to_csv(os.path.join(path_out,'model_in_weekday.csv'), index=False)


# (3) only non commercial
df_cleaned_noncom = df[df['providertype'].str.match('1: consumer')]
df_cleaned_noncom = df_cleaned_noncom.reset_index(drop=True)
# check
print('Number of trips after filter non commercial:',len(df_cleaned_noncom))
# take 250k sample
df_cleaned_noncom_sample = df_cleaned_noncom.sample(n=250000)
df_cleaned_noncom_sample = df_cleaned_noncom_sample.reset_index(drop=True)
# save to disk
df_cleaned_noncom_sample.to_csv(os.path.join(path_out,'model_in_noncom.csv'), index=False)


# (4) only 500m<tripdistancemeter<100km & only weekday trips & only non commercial trips
df_cleaned = df[df['tripdistancemeters'].between(lmin, lmax)]
df_cleaned = df_cleaned[df_cleaned['startdate'].str.match('weekday')]
df_cleaned = df_cleaned[df_cleaned['providertype'].str.match('1: consumer')]
df_cleaned = df_cleaned.reset_index(drop=True)
# check
print('Number of trips after filter all 3:',len(df_cleaned))
# take 250k sample
df_cleaned_sample = df_cleaned.sample(n=250000)
df_cleaned_sample = df_cleaned_sample.reset_index(drop=True)
# save to disk
df_cleaned_sample.to_csv(os.path.join(path_out,'model_in_lb_ub_weekday_noncom.csv'), index=False)






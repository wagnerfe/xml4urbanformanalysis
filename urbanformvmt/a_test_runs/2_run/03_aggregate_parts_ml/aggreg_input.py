"""
Module that aggregates all feature parts in one large dataframe, called df_all_parts_ft

edit on 22.03.21:
only aggregates all street features to one matrix


For each test case we create a 250k sample

@authors: Nikola, Felix W
last modified: 25.02.2021

"""

## Imports
import pandas as pd


## Path to files
path_file = '/p/projects/vwproject/felix_files/data/output/' 
path_out = '/p/projects/vwproject/felix_files/data/output/ml/model_streets_raw.csv' 
# number of parts 
parts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]

# Initialise DataFrames
df_all_parts_blocks = pd.DataFrame()
df_all_parts_streets = pd.DataFrame()
df_all_parts_ft = pd.DataFrame()


## Read in all street features in one dataframe
for part in parts:
    df_new_file = pd.read_csv(path_file + 'berlin_trips_street_dist_ft_part_' + str(part) + '.csv')
    df_all_parts_streets = pd.concat([df_all_parts_streets,df_new_file], ignore_index=True)
    df_all_parts_streets = df_all_parts_streets.drop_duplicates(subset=['tripid'])

# we keep old naming, as this allows to include building and block features in the future
df_all_parts_ft = df_all_parts_streets
# remove block df
del df_all_parts_streets

# save
df_all_parts_ft.to_csv(path_out, index=False)

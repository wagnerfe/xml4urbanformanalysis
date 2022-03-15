"""
Module that aggregates all feature parts in one large dataframe, called df_all_parts_ft

@authors: Nikola, Felix W
last modified: 25.02.2021

"""

# imports
import pandas as pd


#Path to files
path_file = '/data/metab/urbanformvmt/data/output/berlin_trips_parts_features/' 
path_out = '/data/metab/urbanformvmt/data/output/ml_tests/model_input.csv' 

# number of parts 
parts=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30]
#parts=[0,1,2,3]

# Initialise DataFrames
df_all_parts_blocks = pd.DataFrame()
df_all_parts_streets = pd.DataFrame()
df_all_parts_ft = pd.DataFrame()

# Read in all building parts in one dataframe
for part in parts:
    df_new_file = pd.read_csv(path_file + 'berlin_trips_building_bld_dist_ft_part_' + str(part) + '.csv')
    df_all_parts_ft = pd.concat([df_all_parts_ft,df_new_file], ignore_index=True)

# Discard useless columns
df_all_parts_ft.drop(['buffer_part','part','has_buffer'], axis=1, inplace=True)

# Read in all block parts in one dataframe
for part in parts:
    df_new_file = pd.read_csv(path_file + 'berlin_trips_block_dist_ft_part_' + str(part) + '.csv')
    df_all_parts_blocks = pd.concat([df_all_parts_blocks,df_new_file], ignore_index=True)

#Discard useless columns
df_all_parts_blocks.drop(['tripdistancemeters','lengthoftrip','geometry','buffer_part','part','has_buffer'], axis=1, inplace=True)
df_all_parts_ft = df_all_parts_ft.merge(df_all_parts_blocks,left_on = 'tripid', right_on = 'tripid')

# (maybe) discard again columns from merged df (probably “trip_id_right”)
# (maybe) rename tripid_left -> tripid

# remove block df
del df_all_parts_blocks

# Read in all street features in one dataframe
for part in parts:
    df_new_file = pd.read_csv(path_file + 'berlin_trips_street_dist_ft_part_' + str(part) + '.csv')
    df_all_parts_streets = pd.concat([df_all_parts_streets,df_new_file], ignore_index=True)

# Discard useless columns
df_all_parts_streets.drop(['tripdistancemeters','lengthoftrip','geometry','buffer_part','part','has_buffer'], axis=1, inplace=True)
df_all_parts_ft = df_all_parts_ft.merge(df_all_parts_streets,left_on = 'tripid', right_on = 'tripid')

# (maybe) discard again columns from merged df (probably “trip_id_right”)
# (maybe) rename tripid_left -> tripid
# remove block df
del df_all_parts_streets


# save
df_all_parts_ft.to_csv(path_out, index=False)

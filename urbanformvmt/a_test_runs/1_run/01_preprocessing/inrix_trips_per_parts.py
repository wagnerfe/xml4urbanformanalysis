"""
Module that calculates which parts of the inrix data set are inside of Berlin and which are not.
It calculates this for both trip origins and destination

@authors: Nikola, Felix W
last modified: 25.02.2021

"""

# Import Functions

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

##--------------------------------------------------------------##
## Add paths
# get path of current directory
cwd = os.getcwd()
# get back to the path of the root directory
path_module = cwd.split('code')[0]+'code'
# add path to enable the import of the modules
sys.path.append(path_module)

from utils.helpers import import_csv_w_wkt_to_gdf, import_trip_csv_to_gdf


# Set paths and constants
path_to_data_in = '/data/metab/urbanformvmt/data/input'
path_to_data_out = '/data/metab/urbanformvmt/data/output'

path_inrix = os.path.join(path_to_data_in,'tripdata.csv')
path_berlin_boundary = os.path.join(path_to_data_in,'Berlin_boundaries.csv')
path_berlin_boundary_parts = os.path.join(path_to_data_in,'Berlin_boundaries_parts.csv')

crs = 'epsg:25833'

##--------------------------------------------------------------##
## Import Files
# Berlin files
berlin_boundary = import_csv_w_wkt_to_gdf(path_berlin_boundary,crs)
berlin_boundary_parts = import_csv_w_wkt_to_gdf(path_berlin_boundary_parts,crs)

# Inrix files (this time already preprocessed)
gdf_trip_origin, gdf_trip_dest = import_trip_csv_to_gdf(path_inrix,crs=crs)
points = gdf_trip_origin
##--------------------------------------------------------------##
## def function to get data within part
def get_data_within_part(part,points,boundary_parts):
    
        print(part)
        
        part_gdf = boundary_parts[boundary_parts['part'] == part][boundary_parts.has_buffer=='no_buffer']
        part_buffer_gdf = boundary_parts[boundary_parts['part'] == part][boundary_parts.has_buffer=='buffer']
  
        # spatial join layer and part
        df_in_part = gpd.sjoin(points, part_gdf, how='inner', op='within')

        # spatial join layer and part + buffer 
        df_in_part_plus_buffer = gpd.sjoin(points, part_buffer_gdf, how='inner', op='intersects')

        ## get buffered values only
        df_in_buffer_only = df_in_part_plus_buffer[~df_in_part_plus_buffer.index.isin(df_in_part.index)]

        # mark buffered buildings
        df_in_part['index_right'] = False
        df_in_buffer_only['index_right'] = True

        ## append buffered area marked
        df_in_part = df_in_part.append(df_in_buffer_only)

        # change buffer col name
        df_in_part.rename(columns={'index_right':'buffer_part'}, inplace=True)  
        
        return df_in_part

##--------------------------------------------------------------##
## For all parts calculate number of points in part and save in separate csv
for i in range(berlin_boundary_parts.iloc[-1]['part']+1):   
    df_in_part = pd.DataFrame()
    df_in_part = get_data_within_part(i,points,berlin_boundary_parts)
    df_in_part.to_csv(os.path.join(path_to_data_out,'berlin_origin_trips_part_{}.csv'.format(i)), 
                      index=False)
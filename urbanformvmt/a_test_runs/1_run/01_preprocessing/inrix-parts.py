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

# Inrix files
gdf_trip_origin, gdf_trip_dest = import_trip_csv_to_gdf(path_inrix,crs=crs)

##--------------------------------------------------------------##
## Calculate inside and outside trips for ORIGIN
# assign origin trips to points
print(len(gdf_trip_origin))

# create gpd out of Berlin boundary for spatial join
boundary_berlin_geom = gpd.GeoDataFrame(geometry=gpd.GeoSeries(berlin_boundary.iloc[0].geometry),crs=crs)

# get points within Berlin
df_in = gpd.sjoin(gdf_trip_origin,boundary_berlin_geom,how='inner',op='within')
df_out = gdf_trip_origin[~gdf_trip_origin.index.isin(df_in.index)]

# rename and append everyone back together
df_in['index_right'] = True
df_out['index_right'] = False
gdf_trip_origin = df_in.append(df_out)
# rename
gdf_trip_origin.rename(columns={'index_right':'within_berlin'}, inplace=True)  

## Calculate inside and outside trips for DESTINATION
# assign origin trips to points
print(len(gdf_trip_dest))

# get points within Berlin
df_in2 = gpd.sjoin(gdf_trip_dest,boundary_berlin_geom,how='inner',op='within')
df_out2 = gdf_trip_dest[~gdf_trip_dest.index.isin(df_in.index)]

# rename and append everyone back together
df_in2['index_right'] = True
df_out2['index_right'] = False
gdf_trip_dest = df_in2.append(df_out)
# rename
gdf_trip_dest.rename(columns={'index_right':'within_berlin'}, inplace=True)  

##--------------------------------------------------------------##
## Save in separate csv_s
# save on disk
df_in.to_csv(os.path.join(path_to_data_out,'berlin_origin_trips_in.csv'),
                           index=False)

df_out.to_csv(os.path.join(path_to_data_out,'berlin_origin_trips_out.csv'),
                           index=False)

gdf_trip_origin.to_csv(os.path.join(path_to_data_out,'berlin_origin_trips_in_out.csv'),
                           index=False)

# save on disk
df_in2.to_csv(os.path.join(path_to_data_out,'berlin_dest_trips_in.csv'),
                           index=False)

df_out2.to_csv(os.path.join(path_to_data_out,'berlin_dest_trips_out.csv'),
                           index=False)

gdf_trip_dest.to_csv(os.path.join(path_to_data_out,'berlin_dest_trips_in_out.csv'),
                           index=False)


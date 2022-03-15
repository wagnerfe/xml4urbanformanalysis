## Imports
import sys,os
import pandas as pd
import geopandas as gpd
#import mapclassify
import numpy as np
from shapely import wkt


## 1. Define Constants
crs = 'epsg:25833'

## 2. Import dir and functions, paths and data
# add path to ufo map
path_ufo_map = '/data/metab/UFO-MAP'
# add path to enable the import of the modules
sys.path.append(path_ufo_map)
# import functions
from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf, get_data_within_part

# Define paths
path_to_data = '/p/projects/vwproject/felix_files/data/input/'
path_trips = '/p/projects/vwproject/build-environment-traffic/tripdata.csv'
path_berlin_boundary = os.path.join(path_to_data,'boundaries/Berlin_boundaries.csv')
path_berlin_boundary_parts = os.path.join(path_to_data,'boundaries/Berlin_boundaries_parts.csv')
path_out_parts = os.path.join(path_to_data,'tripdata/trip_parts_500k')
path_out_file = os.path.join(path_to_data,'tripdata/trip_500k.csv')

## 3. Read in trips
# Inrix files 
df = pd.read_csv(path_trips)
# Sample data
df_sample = df.sample(n=500000)
df_sample = df_sample.reset_index(drop=True)

# read in start location from csv
gdf_origin = gpd.GeoDataFrame(df_sample, geometry=gpd.points_from_xy(df_sample.startloclon, df_sample.startloclat),crs=crs)
gdf_origin = gdf_origin[['tripid','tripdistancemeters','lengthoftrip','startdate','providertype','geospacialtype','geometry'] ]
# Read in boundary parts
berlin_boundary = import_csv_w_wkt_to_gdf(path_berlin_boundary,crs)
berlin_boundary_parts = import_csv_w_wkt_to_gdf(path_berlin_boundary_parts,crs)

## 4. Data Cleaning
# A) Set lower and upper bounds
#lmin = 500
#lmax = 100000
#gdf_cleaned = gdf_origin[gdf_origin['tripdistancemeters'].between(lmin, lmax)]
#gdf_cleaned = gdf_cleaned.reset_index(drop=True)
#print('Number of trips after bounds:',len(gdf_cleaned))

# B) insert weekday info
gdf_origin['startdate'] = pd.to_datetime(gdf_origin['startdate'])
gdf_origin['startdate'] = np.where(((gdf_origin['startdate']).dt.dayofweek) < 5,'weekday','weekend')
#gdf_cleaned = gdf_cleaned[gdf_cleaned['startdate'].str.match('weekday')]
#gdf_cleaned = gdf_cleaned.reset_index(drop=True)
#print('Number of trips after weekdays:',len(gdf_cleaned))

# C) only non commercial
#gdf_cleaned = gdf_cleaned[gdf_cleaned['providertype'].str.match('1: consumer')]
#gdf_cleaned = gdf_cleaned.reset_index(drop=True)
#print('Number of trips after commercial:',len(gdf_cleaned))

# Sample to size: 250k
#gdf_cleaned_sample = gdf_cleaned.sample(n=250000)
#gdf_cleaned_sample = gdf_cleaned_sample.reset_index(drop=True)

# Print to check
#print('Number of trips after cleaning:',len(gdf_cleaned_sample))

## 5. Save to disk
# Save trip data as one file
gdf_origin.to_csv(path_out_file,index=False)

# Split into 30 parts
# For all parts calculate number of points in part and save in separate csv
for i in range(berlin_boundary_parts.iloc[-1]['part']+1):    
	df_in_part = pd.DataFrame()
	df_in_part = get_data_within_part(i,gdf_origin,berlin_boundary_parts)
	df_in_part.to_csv(os.path.join(path_out_parts,'trip_parts_500k_{}.csv'.format(i)), 
	                  index=False)



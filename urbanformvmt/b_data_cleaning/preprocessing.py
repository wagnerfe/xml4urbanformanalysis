## Imports
import sys,os
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import matplotlib.pyplot as plt


## 1. Define Constants
crs = 25833

## 2. Import dir and functions, paths and data
# 2.1 Add paths
path_root = '/p/projects/vwproject/felix_files/ufo-map'
#path_root = '/Users/Felix/Documents/Studium/PhD/05_Projects/05_UFO-MAP/ufo-map/ufo_map/'
# add path to enable the import of the modules
sys.path.append(path_root)

# 2.2 Set paths
path_trips = '/p/projects/vwproject/ben_files/urban-form/clustering_berlin_2017-2017_eps0.1_minpts5_maxpts1000.csv'
path_boundaries = '/p/projects/vwproject/felix_files/data/input/boundaries/Berlin_boundaries.csv'
path_out = '/p/projects/vwproject/felix_files/data/run_21.07.05/in/trips_all_in_bounds_consumer_weekdays_5-10h_eps0.1_minpts5_maxpts1000.csv'

# 2.3 Import functions
#from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf, get_data_within_part
from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
from cleaning import trip_inside_bounds, tripdistance_bounds, set_weektime

def import_trip_csv_to_gdf_adjust(path,crs):
    '''
    Import trip csv file from Inrix data with WKT geometry column into a GeoDataFrame. 
    We import this here, as the function has to be adjusted according to the input df and column names.

    Last modified: 25/02/2020. By: Felix

    '''
    
    df = pd.read_csv(path)
    # rename columns to work with existing functions
    df = df.rename(columns={'tripdistancekm':'tripdistancemeters'})
    # change km to m
    df.tripdistancemeters = df.tripdistancemeters*1e3
    
    # swap back reversed lat lons from DBSCAN
    df['startloclon_new'] = np.where(df.reversed == False,df.startloclon, df.endloclon)
    df['startloclat_new'] = np.where(df.reversed == False, df.startloclat, df.endloclat)
    df['endloclon_new'] = np.where(df.reversed == False, df.endloclon, df.startloclon)
    df['endloclat_new'] = np.where(df.reversed == False, df.endloclat, df.startloclat)
    df = df.drop(columns={'startloclon','startloclat','endloclon','endloclat'})
    df = df.rename(columns={'startloclon_new':'startloclon','startloclat_new':'startloclat','endloclon_new':'endloclon','endloclat_new':'endloclat'})

    # read in start location from csv
    gdf_origin = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.startloclon, df.startloclat),crs=crs)
    gdf_origin = gdf_origin[['tripid','tripdistancemeters','lengthoftrip','tripmeanspeedkph','tripmaxspeedkph','startdate','enddate','reversed','geometry'] ]
    # read in end location from csv
    gdf_dest = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.endloclon, df.endloclat),crs=crs)
    gdf_dest = gdf_dest[['tripid','tripdistancemeters','lengthoftrip','tripmeanspeedkph','tripmaxspeedkph','startdate','enddate','reversed','geometry'] ]
    
    return (gdf_origin, gdf_dest)

## 3. Read in trips (gdf_o = trip origins, gdf_d trip destinations)
gdf_o, gdf_d = import_trip_csv_to_gdf_adjust(path_trips, 4326)
gdf_o = gdf_o.to_crs(crs)
gdf_d = gdf_d.to_crs(crs)

# Read in Berlin boundaries
gdf_berlin_boundaries = import_csv_w_wkt_to_gdf(path_boundaries,crs)
# Take inner boundary
gdf_berlin_bound = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_berlin_boundaries.iloc[0].geometry),crs=crs)

### FOR TESTING GET ONLY A SAMPLE
#gdf_o = gdf_o.sample(n=10000)
#gdf_o = gdf_o.reset_index(drop=True)

#gdf_d = gdf_d.sample(n=10000)
#gdf_d = gdf_d.reset_index(drop=True)

# Choose for cleaning the data
trips_inside_Berlin = True
trip_bounds = True
weektime = True 
consumer = False
save_file = True


## 4 Clean Data 
if trips_inside_Berlin:
    # 4.1 Take only trips that start and end inside of Berlin 
    gdf_trips = trip_inside_bounds(gdf_o,gdf_d,gdf_berlin_bound)
    print('Number of trips after filtering berlin bounds trips:',len(gdf_trips))
    print(gdf_trips.head())

# 4.2. Take only trips with 500m < tripdistancemeters
if trip_bounds:
    lmin = 500 		                            # in meter
    lmax = max(gdf_trips.tripdistancemeters)	# (no bound) in meter
    gdf_trips = tripdistance_bounds(gdf_trips,lmin,lmax)
    print('Number of trips after filtering triplength trips:',len(gdf_trips))

# 4.3 Take only trips within a specific time and determine weekday/weekend
if weektime:
    weekend = False # All trips should be on Mon, Tue,..,Fr
    tstart = datetime.time(5, 0, 0) # 0am
    tend = datetime.time(10, 0, 0)  # 24pm
    gdf_trips = set_weektime(gdf_trips,weekend, tstart, tend)
    print('Number of trips after filtering weektime trips:',len(gdf_trips))

# 4.4 Take only consumer trips
if consumer:
    gdf_trips = gdf_trips[gdf_trips['providertype'].str.match('1: consumer')]
    gdf_trips = gdf_trips.reset_index(drop=True)
    print('Number of trips after filtering commercial trips:',len(gdf_trips))

## 5. Save to disk
# Save cleaned data as one file
if save_file:
    gdf_trips.to_csv(path_out,index=False)




# Split into 30 parts
# For all parts calculate number of points in part and save in separate csv
#for i in range(berlin_boundary_parts.iloc[-1]['part']+1):    
#	df_in_part = pd.DataFrame()
#	df_in_part = get_data_within_part(i,gdf_origin,berlin_boundary_parts)
#	df_in_part.to_csv(os.path.join(path_out_parts,'trip_parts_500k_{}.csv'.format(i)), 
#	                  index=False)



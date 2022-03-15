"""
Module that calculates which parts of the inrix data set are inside of Berlin and which are not.
It calculates this for both trip origins and destination

@authors: Nikola, Felix W
last modified: 25.02.2021

"""

# import libraries
import pandas as pd
import geopandas as gpd
import os
import sys
import argparse


# get back to the path of the root directory
path_ufo_map = '/data/metab/UFO-MAP'
sys.path.append(path_ufo_map)

path_code = '/p/projects/vwproject/felix_files/urbanformvmt/0_test_runs/2_run'
sys.path.append(path_code)

# import own functions
from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
from utils.features_utils_cluster import features_streets

# argument parser (get number of the part to execute)
parser = argparse.ArgumentParser()
parser.add_argument('-i', help="SLURM_ARRAY_TASK_ID", type=str)
args = parser.parse_args()
part = args.i
print(part)


## Paths 
# path folder files existing Berlin
path_berlin = '/data/metab/osm/learning-from-urban-form-to-predict-building-heights/Data/2-data_preprocessed/Germany/Berlin/Berlin/Berlin_parts'
# path folder trip parts 
path_trip_part = '/p/projects/vwproject/felix_files/data/input/tripdata/trip_parts_500k'
# path parts 
path_intersection_part = os.path.join(path_berlin,'Berlin_intersections_city_part_'+part+'.csv')
path_street_part = os.path.join(path_berlin,'Berlin_streets_city_part_'+part+'.csv')
path_buildings_part =  os.path.join(path_berlin,'Berlin_buildings_city_part_'+part+'.csv')
path_trip_part = os.path.join(path_trip_part,'trip_parts_500k_'+part+'.csv')
# path output
path_out = '/p/projects/vwproject/felix_files/data/output/'+'berlin_trips_street_dist_ft_part_'+part+'.csv'
# set constants
crs = 25833


## Read in files
# Read in one trip part
gdf_trip_origin_part = import_csv_w_wkt_to_gdf(path_trip_part,crs)
# Read in Intersections part csv (with some network metrics but i don't think we use them)
gdf_intersection_part = import_csv_w_wkt_to_gdf(path_intersection_part,crs)
# Read in Street part csv  (with network metrics)
gdf_street_part = import_csv_w_wkt_to_gdf(path_street_part,crs)
# Read in Building part csv (also needed in the function)
gdf_buildings_part = import_csv_w_wkt_to_gdf(path_buildings_part,crs)
 

## Sample Data 
# Sample X% of trips (make sure to use the same seed in both files)
#gdf_trip_origin_part_sample = gdf_trip_origin_part.sample(frac=0.1, random_state=1)
# Reset indices
#gdf_trip_origin_part_sample = gdf_trip_origin_part_sample.reset_index(drop=True)
# edit: no sample needed
gdf_trip_origin_part_sample = gdf_trip_origin_part

##Calculate Features
# For the trip point df part calculate distance-based street and intersection features
gdf_results = features_streets(gdf_trip_origin_part_sample, gdf_street_part, gdf_intersection_part, buffer_sizes=[500,100])
# Merge Results
gdf_trip_origin_part_sample = gdf_trip_origin_part_sample.merge(gdf_results,how='left',left_index=True,right_index=True)

# Save output
geom =  gdf_trip_origin_part_sample.geometry.apply(lambda x: x.wkt)
gdf_trip_origin_part_sample = pd.DataFrame(gdf_trip_origin_part_sample)
gdf_trip_origin_part_sample.geometry = geom
gdf_trip_origin_part_sample.to_csv(path_out, index=False)

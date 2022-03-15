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
import time

start = time.time()

# get back to the path of the root directory
path_ufo_map = '/data/metab/UFO-MAP'
# add path to enable the import of the modules
sys.path.append(path_ufo_map)

# import functions
from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
from ufo_map.Feature_engineering.buildings import features_buildings_distance_based

# argument parser (get number of the part to execute)
parser = argparse.ArgumentParser()
parser.add_argument('-i', help="SLURM_ARRAY_TASK_ID", type=str)
args = parser.parse_args()
part = args.i
print(part)


## Import paths 
# path folder files existing Berlin
path_berlin = '/data/metab/osm/learning-from-urban-form-to-predict-building-heights/Data/2-data_preprocessed/Germany/Berlin/Berlin/Berlin_features'
# path trip part
path_trip_part = '/p/projects/vwproject/felix_files/data/input/tripdata/trip_parts_500k'
# paths part
path_buildings_part =  os.path.join(path_berlin,'Berlin_part_'+part+'_buildings_fts_buildings.csv')
path_trip_part = os.path.join(path_trip_part,'trip_parts_500k_'+part+'.csv')
# path output
path_out_buildings = '/p/projects/vwproject/felix_files/data/output/' + 'berlin_trips_building_bld_dist_ft_part_' + part + '.csv'

# Set constants
crs = 25833


## Read in files
# Read in one trip part
gdf_trip_origin_part = import_csv_w_wkt_to_gdf(path_trip_part,crs)
# Read in building part csv (with buildings and block features)
gdf_buildings_part = import_csv_w_wkt_to_gdf(path_buildings_part, crs)


## Sample
# Sample X% of trips (make sure to use the same seed in both files)
# gdf_trip_origin_part_sample = gdf_trip_origin_part.sample(frac=0.1, random_state=1)
# Reset indices
# gdf_trip_origin_part_sample = gdf_trip_origin_part_sample.reset_index(drop=True)
# edit: no sampling needed
gdf_trip_origin_part_sample = gdf_trip_origin_part
gdf_trip_origin_part_sample = gdf_trip_origin_part_sample.reset_index(drop=True)

## Calculate
# Calculate building distance based features
print('Computing building distance-based features...')
gdf_features_buildings_distance_based = features_buildings_distance_based(gdf_trip_origin_part_sample, gdf_buildings_part, buffer_sizes=[500,100])                                                               
#Merge
gdf_trip_origin_part_sample = gdf_trip_origin_part_sample.merge(gdf_features_buildings_distance_based,how='left',left_index=True,right_index=True)


## Save
# Save output
geom =  gdf_trip_origin_part_sample.geometry.apply(lambda x: x.wkt)
gdf_trip_origin_part_sample = pd.DataFrame(gdf_trip_origin_part_sample)
gdf_trip_origin_part_sample.geometry = geom

gdf_trip_origin_part_sample.to_csv(path_out_buildings, index=False)

end = time.time()
last = divmod(end - start, 60)
print('Done in {} minutes {} seconds'.format(last[0],last[1])) 
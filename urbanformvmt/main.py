import pandas as pd
pd.set_option('display.width', 0)
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from shapely import wkt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from scipy.spatial import cKDTree
from collections import Counter
import h3
import pickle

# get path root
path_root = os.path.normpath(os.getcwd() + os.sep + os.pardir)

# get back to the path of the root directory
path_ufo_map = os.path.join(path_root,'ufo-map')
cluster_ufo_map = "data/metab/UFO_MAP/ufo_map"
sys.path.append(path_ufo_map)
sys.path.append(cluster_ufo_map)

# path_urbanformvmt
path_urbanformvmt = os.path.join(path_root,'urbanformvmt')
sys.path.append(path_urbanformvmt)

# import own functions
from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
from ufo_map.Feature_engineering.city_level import distance_cbd_shortest_dist, distance_local_cbd_shortest_dist, distance_cbd, distance_local_cbd
from ufo_map.Feature_engineering.socio_econ import pop_dens, social_index, transit_dens, income
from ufo_map.Feature_engineering.streets import feature_beta_index
from ufo_map.Feature_engineering.urban_atlas import features_urban_atlas
from b_data_cleaning.cleaning import get_h3_points, get_h3_polygons
from d_statistics_ml.ml import get_pearson, inter_feature, boosted_trees, sweep_feature_importance, boosted_trees_spatial_cv, baseline_spatial_cv,xgb_spatial_cv
from d_statistics_ml.ml_shap import xgb_spatial_cv_shap, feature_selection
from e_utils.utils import delete_hex_on_boundary

# hexagon (area, avg edge length) with index APERTURE_SIZE (units [])
hex_geometry = [(4250546.8477, 1107.712591), 
                (607220.9782429, 418.6760055),
                (86745.8540347, 158.2446558),
                (12392.2648621, 59.810857940),
                (1770.3235517, 22.606379400),
                (252.9033645, 8.544408276),
                (36.1290521, 3.229482772),
                (5.1612932, 1.220629759),
                (0.7373276, 0.461354684),
                (0.1053325, 0.174375668)
               ]


## Functions
def show_dataframe(df, n):
    print("length of df:", len(df))
    print("----------------")
    print(df.head(n))

class TripFeatures():

    def __init__(self):
        # constants
        self.APERTURE_SIZE = None
        self.RUN_NAME = None
        # variables (mostly pd.DataFrames)   
        self.districts = None
        self.global_center = None
        self.district_center = None
        self.graph_nx = None
        self.pop_dens_loc = None
        self.transit=None
        self.urban_atlas = None
        self.income = None
        self.typologies = None
        self.rel_feat = None
        # variables to save outputs
        self.r_square = None
        self.stats_corr = None
        self.stats_pears = None
        self.stats_trees = None
        self.x_mean = None
        self.ale = None
    

    def get_trips(self, filename, crs, reduce_to=None):
        """
        loads trips from filename also adds Point geometry as each 
        starting location, transforms coordinates to decimal
        using self.proj and makes it a GeoDataFrame
        """
        self.trips = pd.read_csv(filename)
        if reduce_to is not None:
            self.trips = self.trips.loc[range(reduce_to)]
        
        if self.trips.columns.str.contains('geometry').any():
            # depending on input csv, create geodataframe from geometry
            self.trips.geometry = self.trips.geometry.apply(wkt.loads)
            self.trips = gpd.GeoDataFrame(self.trips,geometry=self.trips.geometry,crs=crs)

        else:
            # depending on input csv, create geodataframe from lat, lon
            self.trips = gpd.GeoDataFrame(
                self.trips,
                geometry = gpd.points_from_xy(self.trips.startloclon, self.trips.startloclat), crs=crs)


    def get_intersecting_area_from_shapefile(self, gdf, filename:str, colname:str, crs:int=25833, join_geometry:str="h3_hexagon"):
        """
        Loads data from a shapefile and adds the area of the intersection to 'gdf'
        Args:
            gdf: GeoDataFrame with reference geometries where to add areas
            colname: name of the newly created column
            filename: name of the csv
            crs: target crs
            join_geometry: geometry column in gdf to join by
        Returns:
            gdf with an added area column named colname
        """
        sf_gdf = import_csv_w_wkt_to_gdf(filename,crs)

        #sf_gdf = gpd.read_file(filename)
        # Convert to 'crs' if not already
        #if not sf_gdf.crs == crs:
        #    sf_gdf = sf_gdf.to_crs(crs)
        
        # change colname for intersection operation and add backup column
        if "geometry" in gdf.columns and join_geometry != "geometry":
            gdf['geometry_'] = gdf["geometry"]
        gdf["geometry"] = gdf[join_geometry]

        # check if crs's are the same and if not, change and save original crs
        crs_ = None
        if not gdf[join_geometry].crs == crs:
            gdf.geometry = gdf.geometry.to_crs(crs)

        # get intersection
        intersections = gpd.overlay(gdf, sf_gdf, how='intersection')

        # Get area in m²
        intersections[colname] = intersections['geometry'].area
        value = intersections.groupby('hex_id')[colname].agg('sum')
        gdf = pd.merge(gdf, value, on='hex_id', how='left')
        print(str(np.round(np.isnan(gdf[colname]).sum()/len(gdf),2)*100) + " % of the rows do not have an intersection.")
        # Setting nans to 0
        gdf[colname][np.isnan(gdf[colname])] = 0

        # Undo changes
        if "geometry_" in gdf.columns:
            gdf['geometry'] = gdf["geometry_"]
            gdf = gdf.drop(columns="geometry_")

        return gdf
        

    def get_districts(self, colname:str,path_city_bound, APERTURE_SIZE:int, crs:int=None):
        "assigns trip points to hex raster with APERTURESIZE"

        hex_col = 'hex_id'
        self.districts = get_h3_points(self.trips,colname, APERTURE_SIZE, crs)
        #hexagon_tuples = [h3.h3_to_geo_boundary(x) for x in self.districts[hex_col]]

        # Get H3-Hexagon Polygon
        self.districts['h3_hexagon'] = [Polygon(h3.h3_to_geo_boundary(x, geo_json=True)) for x in self.districts[hex_col]]
        self.districts['h3_hexagon'] = gpd.GeoSeries(self.districts['h3_hexagon']).set_crs(4326).to_crs(crs)
        self.APERTURE_SIZE = APERTURE_SIZE
        
        # get boundaries of city
        gdf_city_boundaries = import_csv_w_wkt_to_gdf(path_city_bound,crs=crs)
        gdf_city_bound = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_city_boundaries.iloc[0].geometry),crs=crs)

        # Delete al hexagons that lie on the boundary of the city
        self.districts = delete_hex_on_boundary(self.districts, gdf_city_bound, crs)
        print('Number of hexagons calculated including all trip origins:', len(self.districts))
        
    def get_graph(self, path_graph):
        "loads graph file via pickle so that we can calculate graph functions"
        with open(path_graph, 'rb') as f: 
            self.graph_nx = pickle.load(f)
        
        
    def get_centers(self, path_global_center, path_district_center, crs):
        "loads city_level center coordinates"
        #get global center
        df = pd.read_csv(path_global_center)
        df.geometry = df.geometry.apply(wkt.loads)
        self.global_center = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)

        # get local center
        df = pd.read_csv(path_district_center)
        df.geometry = df.geometry.apply(wkt.loads)
        self.district_center = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)

        
    def get_pop_dens(self, path_pop_dens, APERTURE_SIZE, crs):
        "loads pop_dens data"
        df = pd.read_csv(path_pop_dens)
        df.geometry = df.geometry.apply(wkt.loads)
        self.pop_dens_loc = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
        self.pop_dens_loc = self.pop_dens_loc.drop(columns={'Unnamed: 0','index_right'})

        if APERTURE_SIZE is not None:
            self.pop_dens_loc = get_h3_polygons(self.pop_dens_loc, APERTURE_SIZE, crs)
            
    def get_transit(self,path_transit, APERTURE_SIZE,crs):
        "loads transit data"
        #init
        hex_col = 'hex_id'
        # load transit csv as gdf
        self.transit = import_csv_w_wkt_to_gdf(path_transit,crs)
        # convert points to hex
        if APERTURE_SIZE is not None:
            # 0. convert trip crs to crs=4326
            self.transit = self.transit.to_crs(epsg=4326)
            # 1. convert trip geometry to lat long
            self.transit['lng']= self.transit['geometry'].x
            self.transit['lat']= self.transit['geometry'].y
            # 2. find hexs containing the transit stations
            self.transit[hex_col] = self.transit.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,APERTURE_SIZE),1)
            # 3. aggregate the stations per hex
            self.transit = self.transit.groupby(hex_col).size().to_frame('stops_in_hex').reset_index()
            # 4. find center of hex 
            self.transit['lat'] = self.transit[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
            self.transit['lng'] = self.transit[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
            # 5. convert to gdf
            self.transit = gpd.GeoDataFrame(self.transit, geometry=gpd.points_from_xy(self.transit.lng, self.transit.lat),crs=4326)
            self.transit = self.transit.to_crs(crs)


    def get_social_index(self, path, APERTURE_SIZE, crs):
        "loads social index data"
        df = pd.read_csv(path)
        df.geometry = df.geometry.apply(wkt.loads)
        self.social_index_loc = gpd.GeoDataFrame(df, geometry=df.geometry,crs=crs)

        if APERTURE_SIZE is not None:
            self.social_index_loc = get_h3_polygons(self.social_index_loc, APERTURE_SIZE, crs)                            

    def get_urban_atlas(self,path,APERTURE_SIZE,crs):
        df = pd.read_csv(path)
        df.geometry = df.geometry.apply(wkt.loads)
        # drop roads 
        df = df[df.ft_typology != 'Roads']
        df = df.reset_index(drop=True)
        self.urban_atlas = gpd.GeoDataFrame(df, geometry=df.geometry,crs=crs)
        # get urban atlas classes 
        self.typologies = self.urban_atlas.class_2018.unique()

    def get_income(self, path, APERTURE_SIZE, crs):
        "loads social index data"
        df = pd.read_csv(path)
        df.geometry = df.geometry.apply(wkt.loads)
        self.income_loc = gpd.GeoDataFrame(df, geometry=df.geometry,crs=crs)

        if APERTURE_SIZE is not None:
            self.income_loc = get_h3_polygons(self.income_loc, APERTURE_SIZE, crs)             


    def save_results(self, path_out):
        """
        Saved results to csv 
        """
        self.districts.to_csv(os.path.join(path_out,self.RUN_NAME+'_districts.csv'), 
                      index=False)
        
        if self.stats_pears is not None:
            self.stats_pears.to_csv(os.path.join(path_out,self.RUN_NAME+'_pears_stats.csv'),
                            index=False)

        if self.stats_corr is not None:
            self.stats_corr.to_csv(os.path.join(path_out,self.RUN_NAME+'_inter_corr.csv'),
                            index=False)

        if self.stats_trees is not None:
            self.stats_trees.to_csv(os.path.join(path_out,self.RUN_NAME+'_stats_trees.csv'),
                            index=False)

        if self.x_mean is not None:  
            self.x_mean.to_csv(os.path.join(path_out,self.RUN_NAME+'_feat_dist.csv'),
                            index=False)

        if self.ale is not None:
            self.ale.to_csv(os.path.join(path_out,self.RUN_NAME+'_ale.csv'),
                            index=False) 

def main():
    # define paths for tripdata, global center, district centers
    path_trips = os.path.join(path_root,'data','run_21.07.05','in','trips_lmin_test','trips_cleaned_lmin_2000m.csv')
    path_city_bound = os.path.join(path_root,'data','run_21.07.05','in','Berlin_boundaries.csv')
    path_graph = os.path.join(path_root,'data','run_21.07.05','in','berlin_street_network_graph.gpickle')
    path_global_center = os.path.join(path_root,'data','run_21.07.05','in','berlin_global_cbd.csv')
    path_district_center = os.path.join(path_root,'data','run_21.07.05','in','berlin_district_cbd.csv')
    path_pop_dens = os.path.join(path_root,'data','run_21.07.05','in','gdf_pop_dens_berlin_2018.csv')
    path_transit = os.path.join(path_root,'data','run_21.07.05','in','berlin_transit_stops_cleaned.csv')
    path_social_index = os.path.join(path_root,'data','run_21.07.05','in','social_index_2019_geometries.csv')
    path_urban_atlas = os.path.join(path_root,'data','run_21.07.05','in','ua_2018_v2.csv')
    path_parking_lots = os.path.join(path_root,'data','run_21.07.05','in','parking_lots_berlin_crs25833.csv')
    path_income = os.path.join(path_root,'data','run_21.07.05','in','income_mean_berlin.csv')
    path_out = os.path.join(path_root,'data','run_21.07.05','out')

    # set constants
    features = TripFeatures()
    APERTURE_SIZE = 8 #defines hex raster size
    features.RUN_NAME = 'run_name'+'_hex_'+str(APERTURE_SIZE)
    path_shap = os.path.join(path_out,features.RUN_NAME)

    # read in data
    features.get_trips(filename=path_trips,crs=25833)
    features.get_districts('geometry',path_city_bound,APERTURE_SIZE,crs=25833)
    features.get_graph(path_graph)
    features.get_centers(path_global_center,path_district_center,crs=25833)
    features.get_pop_dens(path_pop_dens,APERTURE_SIZE,crs=25833)
    features.get_social_index(path_social_index,APERTURE_SIZE,crs=25833)
    features.get_transit(path_transit,APERTURE_SIZE,crs=25833)
    features.get_urban_atlas(path_urban_atlas,APERTURE_SIZE,crs=25833)
    features.get_income(path_income,APERTURE_SIZE,crs=25833)
    
    # apply functions
    features.districts = distance_cbd_shortest_dist(features.districts, features.global_center, features.graph_nx)
    features.districts = distance_local_cbd_shortest_dist(features.districts, features.district_center, features.graph_nx)
    features.districts = pop_dens(features.districts,features.pop_dens_loc,"TOT_P_2018",None)
    features.districts = transit_dens(features.districts, features.transit,"stops_in_hex")
    features.districts = social_index(features.districts,features.social_index_loc,['status_index','dynamic_index'])
    features.districts = pd.concat([features.districts,features_urban_atlas(features.districts,features.urban_atlas,[],features.typologies,False,False,True)], axis=1)     
    features.districts = features.get_intersecting_area_from_shapefile(features.districts, path_parking_lots,
                                                               "feature_parking_area", 25833, "h3_hexagon")
    features.districts = income(features.districts,features.income_loc,'weigthed_mean')
    features.districts = feature_beta_index(features.districts, features.graph_nx)
    # show dataframe
    show_dataframe(features.districts,5)

    # Calculate linear statistics
    features.stats_pears = get_pearson(features.districts,'tripdistancemeters')
    features.stats_corr = inter_feature(features.districts, 10, False)
    
    # Calculate Feature Selection on whole sample with all features
    list_drop_features = feature_selection(features.districts,'tripdistancemeters',True,10)
    # Drop unimportant features from features.districts and save in df_ml
    df_ml = features.districts.drop(columns = list_drop_features)
    show_dataframe(df_ml,5)
    
    # Caclulate Prediction on features
    features.stats_trees = xgb_spatial_cv_shap(df_ml,path_shap,'tripdistancemeters',5,True,False,False)
    features.stats_base = baseline_spatial_cv(df_ml,'tripdistancemeters',5,False)

    # Save results in csv
    features.save_results(path_out)
    
if __name__ == "__main__":
    main() 

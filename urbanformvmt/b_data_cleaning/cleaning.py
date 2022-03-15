## Imports
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.ops import cascaded_union
from shapely.geometry import mapping
import h3
import numpy as np
import datetime

## Data Cleaning Functions
def trip_inside_bounds(gdf_o,gdf_d,gdf_bound):
    """
    Function to get all trips that start and end within given boundaries

    Args:
        - gdf_o: geodataframe with trip origin waypoint
        - gdf_d: geodataframe with trip destination waypoints
        - gdf_bounds: geodataframe with boundaries

    Returns:
        - gdf_out: geodataframe with all trips that start and end inside bounds

    Last update: 14/04/21. By Felix.
    """
    # Get only trips that start / end in berlin
    gdf_in_o = gpd.sjoin(gdf_o,gdf_bound,how='inner',op='within')
    gdf_in_d = gpd.sjoin(gdf_d,gdf_bound,how='inner',op='within')
    gdf_in_o = gdf_in_o.drop(columns="index_right")
    gdf_in_d = gdf_in_d.drop(columns="index_right")

    # Merge to one gdf
    gdf_in_d = gdf_in_d.rename(columns={"geometry": "geometry_dest"})
    gdf_in_d = gdf_in_d[['tripid','geometry_dest']]
    gdf_out = gdf_in_o.merge(gdf_in_d,left_on = 'tripid', right_on = 'tripid')
    return gdf_out


def tripdistance_bounds(gdf,lmin,lmax):
	"""
	Function to set upper and lower bounds on tripdistance.
 
    Args:
        - gdf: geodataframe with trip origin waypoint
        - lmin,lmax: min max bounds for trip length

    Returns:
        - gdf_out: geodataframe with lower and upper bounds

    Last update: 13/04/21. By Felix.
	"""
	gdf_out = gdf[gdf['tripdistancemeters'].between(lmin, lmax)]
	gdf_out = gdf_out.reset_index(drop=True)
	return gdf_out



def set_weektime(gdf, weekend, start_hour, end_hour):
    """
    Function to filter for weekdays or weekends.

    Args:
        - gdf: geodataframe with trip origin waypoint
        - weekend (bool): 
            0 := no weekend (Mo,...,Fr)
            1 := weekend (Sat, Sun)
        - start_hour, end_hour (datetime format):
            f.e. 07:00:00 -> datetime.time(7, 0, 0)

    Returns:
        - gdf_out: geodataframe with trips only on either weekdays or weekends
        and only starting between start_hour and end_hour

    Last update: 13/04/21. By Felix.
    """

    ## in whole function
    if weekend:
        gdf['startdate'] = pd.to_datetime(gdf['startdate'])
        gdf = gdf[((gdf['startdate']).dt.dayofweek) >= 5]
        df_hour = gdf.startdate.dt.time
        gdf_out = gdf[(start_hour<=df_hour)&(df_hour<=end_hour)]
    else:
        gdf['startdate'] = pd.to_datetime(gdf['startdate'])
        gdf = gdf[((gdf['startdate']).dt.dayofweek) < 5]
        df_hour = gdf.startdate.dt.time
        gdf_out = gdf[(start_hour<=df_hour)&(df_hour<=end_hour)]

    return gdf_out


def get_h3_points(gdf, colname:str, APERTURE_SIZE:int=8, crs:int=25833):
    """
    Function that maps all trip points on a hex grid and normalises trip numbers per
    hexagon.

    Args:
        - gdf: dataframe with cleaned trip origin waypoints
        - APERTURE_SIZE: hex raster; for more info see: https://h3geo.org/docs/core-library/restable/ 
        - crs: crs of input gdf

    Returns:
        - gdf_out: geodataframe with hexagons containing the average trip lengths & duration per hexagon

    Last update: 15/04/21. By Felix.
    """
    #hex_col = 'hex'+str(APERTURE_SIZE)
    hex_col = 'hex_id'

    # take colname as geometry col (default:'geometry', else:'geometry_dest') 
    if not colname == 'geometry':
        #print(gdf[colname].head())
        #print(type(gdf[colname]))
        #gdf['geometry'] = gpd.GeoSeries.from_wkt(gdf[colname]).set_crs(crs)
        gdf['geometry'] = gpd.GeoSeries(gdf[colname].apply(wkt.loads),crs=crs)
        print('transfered geometry_dest successfully')

    #convert crs to crs=4326
    gdf = gdf.to_crs(epsg=4326)
    
    # 0. convert trip geometry to lat long
    gdf['lng']= gdf['geometry'].x
    gdf['lat']= gdf['geometry'].y

    # 0. find hexs containing the points
    gdf[hex_col] = gdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,APERTURE_SIZE),1)
    
    # 1. group all trips per hexagon and average tripdistancemters
    df_out = gdf.groupby(hex_col)['tripdistancemeters'].mean().to_frame('tripdistancemeters').reset_index()
    
    # 2. calculate average trip duration per hex
    df_length_of_trip = gdf.groupby(hex_col)['lengthoftrip'].mean().to_frame('lengthoftrip').reset_index()
    df_out['lengthoftrip'] = df_length_of_trip.lengthoftrip
    
    # 3. count number of trips per hex
    df_cnt = gdf.groupby(hex_col).size().to_frame('cnt').reset_index()
    df_out['points_in_hex'] = df_cnt.cnt  
    
    # 4. Get center of hex to calculate new features
    df_out['lat'] = df_out[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    df_out['lng'] = df_out[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])   

    # 5. Convert lat and long to geometry column 
    gdf_out = gpd.GeoDataFrame(df_out, geometry=gpd.points_from_xy(df_out.lng, df_out.lat),crs=4326)
    gdf_out = gdf_out[[hex_col,'tripdistancemeters','lengthoftrip','points_in_hex','geometry']]

    # 6. Convert to crs
    gdf_out = gdf_out.to_crs(crs)
    
    return gdf_out



def get_h3_polygons(gdf,APERTURE_SIZE,crs):
    """
    Function to transform polygon data into h3 grid with given aperture size
    
    Args:
        - gdf: dataframe with data provided in polygons
        - APERTURE_SIZE: hex raster; for more info see: https://h3geo.org/docs/core-library/restable/ 
        - crs: crs of input gdf

    Returns:
        - gdf_out: geodataframe 'gdf' and 2 additional columns, containing hexagon IDs and point geometry

    Last update: 15/04/21. By Felix.
    """
    # transform input gdf to right crs
    gdf = gdf.to_crs(4326) 
    # Unify the CT boundries
    union_poly = cascaded_union(gdf.geometry)
    # Find the hexs within the city boundary using PolyFill
    hex_list=[]
    for n,g in enumerate(union_poly):
        #print(n,'\r')
        temp  = mapping(g)
        temp['coordinates']=[[[j[1],j[0]] for j in i] for i in temp['coordinates']]  
        hex_list.extend(h3.polyfill(temp,APERTURE_SIZE))

    # create hex dataframe
    #hex_col = 'hex{}'.format(APERTURE_SIZE)
    hex_col = 'hex_id'
    dfh = pd.DataFrame(hex_list,columns=[hex_col])
    print('Sanity Check\nnumber of hexes:', len(hex_list))
    print('number of duplicates:', len(hex_list) - len(dfh.drop_duplicates()))

    # add lat & lng of center of hex 
    dfh['lat']=dfh[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    dfh['lng']=dfh[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
    dfh = gpd.GeoDataFrame(dfh,geometry=gpd.points_from_xy(dfh.lng, dfh.lat),crs="epsg:4326")
    
    # Intersect Hex Point with gdf Polygons 
    gdf_out = gpd.sjoin(gdf,dfh, how="right")
    gdf_out = gdf_out.drop(columns={"index_left","lat","lng"})

    # transform into predefined crs
    gdf_out = gdf_out.to_crs(crs)

    return gdf_out

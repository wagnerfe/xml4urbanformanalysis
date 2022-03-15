import geopandas as god
import psutil
from shapely.geometry import Polygon



def get_indexes_right_round_buffer(gdf,building_gdf,buffer_size):

        # Create a gdf with just building geometries (so that we do carry around all the columns in the future joints)
        building_gdf_for_join = gpd.GeoDataFrame(geometry=building_gdf.geometry)

        # Create a gdf with a buffer per object of interest as a single geometry column
        buffer = gdf.geometry.centroid.buffer(buffer_size).values
        buffer_gdf = gpd.GeoDataFrame(geometry=buffer)

        # Join each buffer with the buildings that intersect it, as a gdf
        # where each row is a pair buffer (index) and building (index_right)
        # there are multiple rows for one index
        joined_gdf = gpd.sjoin(buffer_gdf, building_gdf_for_join, how="left", op="intersects")
        print(psutil.virtual_memory())

        # Remove rows of the building of interest for each buffer
        joined_gdf = joined_gdf[joined_gdf.index != joined_gdf.index_right]

        return(buffer,joined_gdf)


def get_indexes_right_bbox(geometries,gdf_inter_sindex,buffer_size,small_mode=False,longuest_axes=None):
    """ Get the indexes of the objects within a buffer.
    (buildings, streets, land use, sbb) 

    With small_mode, one can exclude object within a distance to the bounding
    box, to save time later in the function. This is meant to be used for 
    buildings only at the moment.

    Args:
        geometries (list): list of geometries from gdf (points)
        gdf_inter_sindex (RTree index): sindex of gdf_intersect (e.g. ua in features_urban_atlas)
        buffer_size (int or list): if int it defines the buffer size, 
                                   if list it contains a list of polygons define the buffer for each geom
        smal_mode (bool): one can exclude object within a distance to the bounding box, to save time later in the function. 
                                   This is meant to be used for buildings only at the moment.
        longuest_axes (bool): ???
    Out:
        indexes_right (list): List of lists indicating which gdf_intersect with wich gdf
        bbox_geom (list): list of bbox geometries

    last modified by: Felix on 2021/07/13
    """

    # initialize output lists
    bbox_geom = [None]*len(geometries)
    indexes_right = [None]*len(geometries)
    if small_mode:
        indexes_right_small = [None]*len(geometries)    
    
    for index,geometry in enumerate(geometries):
        # check if buffer_size is list of hex geoms
        if isinstance(buffer_size,list):
            bbox = buffer_size[index].bounds
            bbox_geom[index] = buffer_size[index]
            indexes_right[index] = list(gdf_inter_sindex.intersection(bbox))
        # otherwise (if buffer_size is int)
        else:    
            bbox = geometry.centroid.buffer(buffer_size).bounds
            # store indexes of buildings in buffer
            indexes_right[index] = list(gdf_inter_sindex.intersection(bbox))

            if small_mode:
                indexes_right_small[index] = inter_small_bbox(longuest_axes,
                                                            indexes_right,
                                                            index,
                                                            buffer_size,
                                                            bbox,
                                                            gdf_inter_sindex)
                
            # get the polygon of each bounding box buffer for 
            # those buildings we need to intersect
            bbox_geom[index] = (Polygon([(bbox[0],bbox[1]),(bbox[0],bbox[3]),\
                (bbox[2],bbox[3]),(bbox[2],bbox[1])]))

    #print(geometry.centroid.buffer(buffer_size).bounds)
    if small_mode:
        return(indexes_right,indexes_right_small,bbox_geom)
    else:
        return(indexes_right,bbox_geom)



def inter_small_bbox(longuest_axes,
                    indexes_right,
                    index,
                    buffer_size,
                    bbox,
                    gdf_inter_sindex):

        la = longuest_axes.iloc[indexes_right[index]].max() * 1.2

        # for computing the area within bbox, we need to remove some part of buildings on the sides
        # get the buildings that would do not want to intersect
        # because they are a least 1.2x the size of the longest axis any the longuest
        # (extra 0.2 for safety if there is some angle)
        if la < buffer_size:
            bbox_small = (bbox[0]+la,bbox[1]+la,bbox[2]-la,bbox[3]-la)
            return(list(gdf_inter_sindex.intersection(bbox_small)))

        else: 
            return([])
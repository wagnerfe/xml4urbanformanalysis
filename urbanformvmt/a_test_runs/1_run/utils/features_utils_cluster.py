# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:37:57 2020

@author: miln
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import time
import osmnx as ox
import networkx as nx
from shapely import wkt
from shapely.geometry import Polygon,MultiPolygon
from shapely.ops import cascaded_union
import math
import random
from collections import Counter

##############################################
##############################################
## THESE ARE FUNCTIONS COPY-PASTED FROM 
## THE LIBRARY MOMEPY, BECAUSE THEY 
## REQUIRE INTERNET ACCESS, MOMEPY CANNOT
## BE USED DIRECTLY ON THE CLUSTER.
## FOR THE RELEASE, CREATE A COPY WITH 
## ORIGINALS FUNCTION.
##
##
##              (START MOMEPY)
##
##############################################
##############################################

def _azimuth(point1, point2):
    """azimuth between 2 shapely points (interval 0 - 180)"""
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

# calculate the radius of circumcircle
def _longest_axis(points):
    circ = _make_circle(points)
    return circ[2] * 2

def _make_circle(points):
    # Convert to float and randomize order
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)

    # Progressively add points to circle or recompute circle
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not _is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c


# One boundary point known
def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not _is_in_circle(c, q):
            if c[2] == 0.0:
                c = _make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c

# Two boundary points known
def _make_circle_two_points(points, p, q):
    circ = _make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q

    # For each point not in the two-point circle
    for r in points:
        if _is_in_circle(circ, r):
            continue

        # Form a circumcircle and classify it on left or right side
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = _make_circumcircle(p, q, r)
        if c is None:
            continue
        elif cross > 0.0 and (
            left is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            > _cross_product(px, py, qx, qy, left[0], left[1])
        ):
            left = c
        elif cross < 0.0 and (
            right is None
            or _cross_product(px, py, qx, qy, c[0], c[1])
            < _cross_product(px, py, qx, qy, right[0], right[1])
        ):
            right = c

    # Select which circle to return
    if left is None and right is None:
        return circ
    if left is None:
        return right
    if right is None:
        return left
    if left[2] <= right[2]:
        return left
    return right

def _make_circumcircle(p0, p1, p2):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ox = (min(ax, bx, cx) + max(ax, bx, cx)) / 2.0
    oy = (min(ay, by, cy) + max(ay, by, cy)) / 2.0
    ax -= ox
    ay -= oy
    bx -= ox
    by -= oy
    cx -= ox
    cy -= oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = (
        ox
        + (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        )
        / d
    )
    y = (
        oy
        + (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        )
        / d
    )
    ra = math.hypot(x - p0[0], y - p0[1])
    rb = math.hypot(x - p1[0], y - p1[1])
    rc = math.hypot(x - p2[0], y - p2[1])
    return (x, y, max(ra, rb, rc))


def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def _is_in_circle(c, p):
    return (
        c is not None
        and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON
    )

def _make_diameter(p0, p1):
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r0 = math.hypot(cx - p0[0], cy - p0[1])
    r1 = math.hypot(cx - p1[0], cy - p1[1])
    return (cx, cy, max(r0, r1))




class momepy_Perimeter:
    """
    Calculates perimeter of each object in given GeoDataFrame. It can be used for any
    suitable element (building footprint, plot, tessellation, block).
    It is a simple wrapper for geopandas `gdf.geometry.length` for the consistency of momepy.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    Examples
    --------
    >>> buildings = gpd.read_file(momepy.datasets.get_path('bubenec'), layer='buildings')
    >>> buildings['perimeter'] = momepy.Perimeter(buildings).series
    >>> buildings.perimeter[0]
    137.18630991119903
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self.series = self.gdf.geometry.length


class momepy_LongestAxisLength:
    """
    Calculates the length of the longest axis of object.
    Axis is defined as a diameter of minimal circumscribed circle around the convex hull.
    It does not have to be fully inside an object.
    .. math::
        \\max \\left\\{d_{1}, d_{2}, \\ldots, d_{n}\\right\\}
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    Examples
    --------
    >>> buildings['lal'] = momepy.LongestAxisLength(buildings).series
    >>> buildings.lal[0]
    40.2655616057102
    """

    def __init__(self, gdf):
        self.gdf = gdf
        hulls = gdf.geometry.convex_hull
        self.series = hulls.apply(lambda hull: _longest_axis(hull.exterior.coords))



class momepy_Elongation:
    """
    Calculates elongation of object seen as elongation of its minimum bounding rectangle.
    .. math::
        {{p - \\sqrt{p^2 - 16a}} \\over {4}} \\over {{{p} \\over {2}} - {{p - \\sqrt{p^2 - 16a}} \\over {4}}}
    where `a` is the area of the object and `p` its perimeter.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    Attributes
    ----------
    e : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    References
    ----------
    Gil J, Montenegro N, Beirão JN, et al. (2012) On the Discovery of
    Urban Typologies: Data Mining the Multi-dimensional Character of
    Neighbourhoods. Urban Morphology 16(1): 27–40.
    Examples
    --------
    >>> buildings_df['elongation'] = momepy.Elongation(buildings_df).series
    >>> buildings_df['elongation'][0]
    0.9082437463675544
    """

    def __init__(self, gdf):
        self.gdf = gdf

        # TODO: vectorize minimum_rotated_rectangle after pygeos implementation
        bbox = gdf.geometry.apply(lambda g: g.minimum_rotated_rectangle)
        a = bbox.area
        p = bbox.length
        cond1 = p ** 2
        cond2 = 16 * a
        bigger = cond1 >= cond2
        sqrt = np.empty(len(a))
        sqrt[bigger] = cond1[bigger] - cond2[bigger]
        sqrt[~bigger] = 0

        # calculate both width/length and length/width
        elo1 = ((p - np.sqrt(sqrt)) / 4) / ((p / 2) - ((p - np.sqrt(sqrt)) / 4))
        elo2 = ((p + np.sqrt(sqrt)) / 4) / ((p / 2) - ((p + np.sqrt(sqrt)) / 4))

        # use the smaller one (e.g. shorter/longer)
        res = np.empty(len(a))
        res[elo1 <= elo2] = elo1[elo1 <= elo2]
        res[~(elo1 <= elo2)] = elo2[~(elo1 <= elo2)]

        self.series = pd.Series(res, index=gdf.index)




class momepy_Corners:
    """
    Calculates number of corners of each object in given geoDataFrame.
    Uses only external shape (shapely.geometry.exterior), courtyards are not included.
    .. math::
        \\sum corner
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    Examples
    --------
    >>> buildings_df['corners'] = momepy.Corners(buildings_df).series
    100%|██████████| 144/144 [00:00<00:00, 1042.15it/s]
    >>> buildings_df.corners[0]
    24
    """

    def __init__(self, gdf):
        self.gdf = gdf

        # define empty list for results
        results_list = []

        # calculate angle between points, return true or false if real corner
        def _true_angle(a, b, c):
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            if np.degrees(angle) <= 170:
                return True
            if np.degrees(angle) >= 190:
                return True
            return False

        # fill new column with the value of area, iterating over rows one by one
        for geom in gdf.geometry:

            if type(geom) == Polygon: # <<<< Modif

                corners = 0  # define empty variables
                points = list(geom.exterior.coords)  # get points of a shape
                stop = len(points) - 1  # define where to stop
                for i in np.arange(
                    len(points)
                ):  # for every point, calculate angle and add 1 if True angle
                    if i == 0:
                        continue
                    elif i == stop:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue

                    else:
                        a = np.asarray(points[i - 1])
                        b = np.asarray(points[i])
                        c = np.asarray(points[i + 1])

                        if _true_angle(a, b, c) is True:
                            corners = corners + 1
                        else:
                            continue

            if type(geom) == MultiPolygon: # <<<< Modif (all if loop added)

                corners = 0  

                for subgeom in geom: 

                    points = list(subgeom.exterior.coords)  # get points of a shape
                    stop = len(points) - 1  # define where to stop
                    for i in np.arange(
                        len(points)
                    ):  # for every point, calculate angle and add 1 if True angle
                        if i == 0:
                            continue
                        elif i == stop:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue

                        else:
                            a = np.asarray(points[i - 1])
                            b = np.asarray(points[i])
                            c = np.asarray(points[i + 1])

                            if _true_angle(a, b, c) is True:
                                corners = corners + 1
                            else:
                                continue

            results_list.append(corners)

        self.series = pd.Series(results_list, index=gdf.index)


class momepy_Convexeity:
    """
    Calculates convexeity index of each object in given geoDataFrame.
    .. math::
        area \\over \\textit{convex hull area}
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects
    areas : str, list, np.array, pd.Series (default None)
        the name of the dataframe column, np.array, or pd.Series where is stored area value. If set to None, function will calculate areas
        during the process without saving them separately.
    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    areas : Series
        Series containing used area values
    References
    ----------
    Dibble, J. (2016) Urban Morphometrics: Towards a Quantitative Science of Urban
    Form. University of Strathclyde.
    Examples
    --------
    >>> buildings_df['convexeity'] = momepy.Convexeity(buildings_df).series
    >>> buildings_df.convexeity[0]
    0.8151964258521672
    """

    def __init__(self, gdf, areas=None):
        self.gdf = gdf

        gdf = gdf.copy()

        if areas is None:
            areas = gdf.geometry.area
        if not isinstance(areas, str):
            gdf["mm_a"] = areas
            areas = "mm_a"
        self.areas = gdf[areas]
        self.series = gdf[areas] / gdf.geometry.convex_hull.area




class momepy_Orientation:
    """
    Calculate the orientation of object
    Captures the deviation of orientation from cardinal directions.
    Defined as an orientation of the longext axis of bounding rectangle in range 0 - 45.
    Orientation of LineStrings is represented by the orientation of line
    connecting first and last point of the segment.
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing objects to analyse
    Attributes
    ----------
    series : Series
        Series containing resulting values
    gdf : GeoDataFrame
        original GeoDataFrame
    References
    ----------
    Schirmer PM and Axhausen KW (2015) A multiscale classiﬁcation of urban morphology.
    Journal of Transport and Land Use 9(1): 101–130. (adapted)
    Examples
    --------
    >>> buildings_df['orientation'] = momepy.Orientation(buildings_df).series
    100%|██████████| 144/144 [00:00<00:00, 630.54it/s]
    >>> buildings_df['orientation'][0]
    41.05146788287027
    """

    def __init__(self, gdf):
        self.gdf = gdf
        # define empty list for results
        results_list = []

        def _dist(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])

        for geom in gdf.geometry:
            if geom.type in ["Polygon", "MultiPolygon", "LinearRing"]:
                # TODO: vectorize once minimum_rotated_rectangle is in geopandas from pygeos
                bbox = list(geom.minimum_rotated_rectangle.exterior.coords)
                axis1 = _dist(bbox[0], bbox[3])
                axis2 = _dist(bbox[0], bbox[1])

                if axis1 <= axis2:
                    az = _azimuth(bbox[0], bbox[1])
                else:
                    az = _azimuth(bbox[0], bbox[3])
            elif geom.type in ["LineString", "MultiLineString"]:
                coords = geom.coords
                az = _azimuth(coords[0], coords[-1])
            else:
                results_list.append(np.nan)

            if 90 > az >= 45:
                diff = az - 45
                az = az - 2 * diff
            elif 135 > az >= 90:
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            elif 181 > az >= 135:
                diff = az - 135
                az = az - 2 * diff
                diff = az - 90
                az = az - 2 * diff
                diff = az - 45
                az = az - 2 * diff
            results_list.append(az)

        self.series = pd.Series(results_list, index=gdf.index)


##############################################
##############################################
## THESE ARE FUNCTIONS COPY-PASTED FROM 
## THE LIBRARY MOMEPY, BECAUSE THEY 
## REQUIRE INTERNET ACCESS, MOMEPY CANNOT
## BE USED DIRECTLY ON THE CLUSTER.
## FOR THE RELEASE, CREATE A COPY WITH 
## ORIGINALS FUNCTION.
##
##
##              (END MOMEPY)
##
##############################################
##############################################


## BUILDING LEVEL ---------------------- 


def features_building_level (df,
                    FootprintArea=True,
                    Perimeter=True,
                    Phi=True,
                    LongestAxisLength=True,
                    Elongation=True,
                    Convexity=True,
                    Orientation=True,
                    Corners=True,
                    Touches=True):
 
    """
    Returns a DataFrame with building-level features.
    
    Extensively uses Momepy: http://docs.momepy.org/en/stable/api.html
    
    All features computed by default.


    Features:
    ---------
    - FootprintArea
    - Perimeter
    - Phi
    - LongestAxisLength
    - Elongation
    - Convexity
    - Orientation
    - Corners
    - TouchesCount
    - SharedWallLength

    """

    # invalid_geoms = []
    
    # for index,row in df.iterrows():
    #     if row.geometry.is_valid == False:
    #         invalid_geoms.append(index)


    # if invalid_geoms != []:
    #     print(invalid_geoms)

    # df = df.drop(invalid_geoms)

    # Create empty result DataFrame
    df_results = pd.DataFrame()
    
    
    if FootprintArea==True:

        print('FootprintArea...')
        
        df_results['FootprintArea'] = df.geometry.area


    if Perimeter==True:

        print('Perimeter...')
        
        df_results['Perimeter'] = momepy_Perimeter(df).series
    
    
    if Phi==True: 

        print('Phi...')

        Phi = []
        
        for index,row in df.iterrows():

            print(index)

            if type(row.geometry) == Polygon:

                # Draw exterior ring of the geom
                exterior = row.geometry.exterior

            if type(row.geometry) == MultiPolygon:

                # Draw exterior ring around largest geom of the multipoly 
                exterior = max(row.geometry, key=lambda a: a.area).exterior

            # Compute max distance to a point
            max_dist = row.geometry.centroid.hausdorff_distance(exterior)

            # Draw smallest circle around building
            min_circle = row.geometry.centroid.buffer(max_dist)

            # Area of the circle
            circle_area = min_circle.area

            # Compute phi
            phi_i = row.geometry.area/circle_area
            
            Phi.append(phi_i)
    
        df_results['Phi'] = Phi

    
    if LongestAxisLength==True:

        print('LongestAxisLength...')
        
        df_results['LongestAxisLength'] = momepy_LongestAxisLength(df).series 
    
    
    if Elongation==True:

        print('Elongation...')
        
        df_results['Elongation'] = momepy_Elongation(df).series
    
    
    if Convexity==True:

        print('Convexity...')
        
        df_results['Convexity'] = momepy_Convexeity(df).series
    
    
    if Orientation==True:

        print('Orientation...')
        
        df_results['Orientation'] = momepy_Orientation(df).series
    
    
    if Corners==True:

        print('Corners...')

        
        df_results['Corners'] = momepy_Corners(df).series
    
    
    if Touches==True: #CountTouches & SharedWallLength

        print('CountTouches and SharedWallLength')

        CountTouches = []    
        SharedWallLength = []

        # Create a spatial index for the OSM building layer
        spatial_index = df.sindex

        for index,row in df.iterrows():

            try:

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(spatial_index.intersection(row.geometry.bounds))

                # Retrieve possible matches
                possible_matches = df.iloc[possible_matches_index] 

                # Identify precise matches that intersect
                precise_matches = possible_matches[possible_matches.intersects(row.geometry)]

                # Store number of buildings touching buildings i
                if len(precise_matches) > 1:

                    # Remove building i for count
                    CountTouches.append(len(precise_matches)-1)

                else:

                    # There is no touching building
                    CountTouches.append(0)    

                # Compute the shared length
                if len(precise_matches) > 1:

                    # Initialize counter for building i
                    shared_length_i = 0


                    # Compute and sum the shared walls length over all matches 
                    for j in range(len(precise_matches)):

                        # Compute the overlap between buildings i and j
                        inter_area = precise_matches.iloc[j]['geometry'].intersection(row.geometry).area

                        # If they have the same area = they are actually the same, pass
                        if inter_area/df_results['FootprintArea'][index] > 0.999:

                            pass

                        else:            

                            # Retrieve matching segment
                            segment_j = precise_matches.iloc[j]['geometry'].intersection(row.geometry)

                            # Add the length to the counter for building i
                            shared_length_i += segment_j.length

                    # Store the shared wall length with building i
                    SharedWallLength.append(shared_length_i)

                else:

                    SharedWallLength.append(0)

            except:
                print('Building {} or a touching one seems to have a topological problem.'.format(index))
                CountTouches.append(np.nan)    
                SharedWallLength.append(np.nan)
                
        
        df_results['CountTouches'] = CountTouches        
        df_results['SharedWallLength'] = SharedWallLength 
                
    
    return(df_results)





## BLOCK LEVEL -------------------------


def features_block_level(df, bloc_features=True):
    """
    Returns a DataFrame with blocks of adjacent buildings and related features.

    Features can be enabled or disabled.

    Non-Feature:
    -----------
    - TouchesIndexes: List of the indexes of the buildings in block.

    Features:
    ---------
    - BlockLength
    - AvBlockFootprintArea
    - StBlockFootprintArea
    - BlockTotalFootprintArea
    - BlockPerimeter
    - BlockLongestAxisLength
    - BlockElongation
    - BlockConvexity
    - BlockOrientation
    - BlockCorners

    """
    # invalid_geoms = []
    
    # for index,row in df.iterrows():
    #     if row.geometry.is_valid == False:
    #         invalid_geoms.append(index)


    # if invalid_geoms != []:
    #     print(invalid_geoms)

    # df = df.drop(invalid_geoms)

    # Create empty result DataFrame
    df_results = pd.DataFrame()

    # Create a spatial index
    df_spatial_index = df.sindex

    # Create empty list 
    TouchesIndexes = []


    ## RETRIEVE BLOCKS
    print('Retrieve blocks')

    for index,row in df.iterrows():

        # Case 1: the block has already been done
        already_in = [TouchesIndex for TouchesIndex in TouchesIndexes if index in TouchesIndex]

        if already_in != []:

            TouchesIndexes.append(already_in[0])

        else:

            try:

                # check if detached building
                possible_touches_index = list(df_spatial_index.intersection(row.geometry.bounds))
                possible_touches = df.iloc[possible_touches_index] 
                precise_touches = possible_touches[possible_touches.intersects(row.geometry)]

            except:

                print('issue')
                TouchesIndexes.append([index])
                continue

            # Case 2: it is a detached building
            if len(precise_touches)==1:
                TouchesIndexes.append([index])

            # Case 3: the block is yet to be done
            else:

                try:

                    current_index = index

                    # lists output
                    # buildings that have already been visited
                    visited = []
                    # directions that need to be explored (direction = index of a touching building)
                    dir_to_explore = []

                    # initiliaze stop
                    it_is_over = False
                    #n= 0

                    while it_is_over != True:
                    #while n < 20:
                        #n +=1    

                        # update index
                        current_building = df.loc[current_index]

                        # touch all buildings around current building
                        possible_touches_index = list(df_spatial_index.intersection(current_building.geometry.bounds))
                        possible_touches = df.iloc[possible_touches_index] 
                        precise_touches = possible_touches[possible_touches.intersects(current_building.geometry)]
                        
                        # add current building to the list of buildings already visited
                        visited.append(current_building.name)

                        # retrieve indices of buildings touching the current building
                        touches_index = precise_touches.index.to_list()
                        
                        # retrieve the touching buildings that have been previously visited 
                        outs_visited = [touch_index for touch_index in touches_index if touch_index in visited]

                        # retrieve the touching buildings that are already listed as direction to explore
                        outs_explore = [touch_index for touch_index in touches_index if touch_index in dir_to_explore]

                        # remove previously visited buildings from the index list
                        for out in range(len(outs_visited)):
                            touches_index.remove(outs_visited[out])    
                            
                        # remove already listed next buildings from the index list
                        for out in range(len(outs_explore)):
                            touches_index.remove(outs_explore[out]) 
                            

                        # decide what is next
                        if len(touches_index) == 0:
                            try:
                                # update from last in memory
                                current_index = dir_to_explore[-1]
                                # 
                                dir_to_explore = dir_to_explore[:-1]        

                            except:
                                # there are no more building in the block
                                it_is_over = True

                        elif len(touches_index) == 1:
                            # update
                            current_index = touches_index[0]

                        else:
                            # update
                            current_index = touches_index[0]
                            # add to memory remaining building
                            dir_to_explore += touches_index[1:]

                    TouchesIndexes.append(visited)

                except:
                    print('issue')
                    TouchesIndexes.append([index])


    df_results['TouchesIndexes'] = TouchesIndexes


    ## COMPUTE METRICS

    if bloc_features==True:

        BlockLength = [None] * len(df)
        AvBlockFootprintArea = [None] * len(df)
        StBlockFootprintArea = [None] * len(df)
        SingleBlockPoly = [None] * len(df)
        BlockTotalFootprintArea = [None] * len(df)

        ## Invidual buildings within block
        print('Manipulate blocks')

        for index,row in df_results.iterrows():

            try:

                # If detached house 
                if row['TouchesIndexes']==[index]:

                    # Append house values:
                    BlockLength[index] = 1
                    AvBlockFootprintArea[index] = df.geometry[index].area
                    StBlockFootprintArea[index] = 0
                    SingleBlockPoly[index] = df.geometry[index]
                    BlockTotalFootprintArea[index] = df.geometry[index].area 

                else:
                    
                    ## block length
                    BlockLength[index] = len(row['TouchesIndexes'])
                    
                    # retrieve block
                    block = df[df.index.isin(row['TouchesIndexes'])]

                    ## Compute distribution individual buildings
                    AvBlockFootprintArea[index] = block.geometry.area.mean()
                    StBlockFootprintArea[index] = block.geometry.area.std()

                    # more?

                    # merge block into one polygon
                    SingleBlockPoly[index] = cascaded_union(block.geometry)

                    # Compute total area
                    BlockTotalFootprintArea[index] = cascaded_union(block.geometry).area

            except:

                    BlockLength[index] = np.nan
                    AvBlockFootprintArea[index] = np.nan
                    StBlockFootprintArea[index] = np.nan
                    SingleBlockPoly[index] = df.geometry[index]
                    BlockTotalFootprintArea[index] = np.nan


        df_results['BlockLength'] = BlockLength
    
        print('Features distribution buildings within block...')

        df_results['AvBlockFootprintArea'] = AvBlockFootprintArea
        df_results['StdBlockFootprintArea'] = StBlockFootprintArea

        ## Whole Block

        print('Features for the whole block...')

        df_results['BlockTotalFootprintArea'] = BlockTotalFootprintArea

        # Momepy expects a GeoDataFrame
        SingleBlockPoly = gpd.GeoDataFrame(geometry=SingleBlockPoly)

        # Compute Momepy building-level features for the whole block
        df_results['BlockPerimeter'] = momepy_Perimeter(SingleBlockPoly).series
        df_results['BlockLongestAxisLength'] = momepy_LongestAxisLength(SingleBlockPoly).series 
        df_results['BlockElongation'] = momepy_Elongation(SingleBlockPoly).series
        df_results['BlockConvexity'] = momepy_Convexeity(SingleBlockPoly).series
        df_results['BlockOrientation'] = momepy_Orientation(SingleBlockPoly).series
        try:
            df_results['BlockCorners'] = momepy_Corners(SingleBlockPoly).series
        except:
            "meh"

    df_results = df_results.fillna(0)

    return(df_results)


def features_buildings_distance_based(df,
    buffer_sizes=None,
    subset_mode = False,
    subset = None,
    block_based=True,
    osm_mode = False):
    """
    Returns a DataFrame with features about the buildings and blocks surrounding each building
    within given distances (circular buffers).

    Block-based features can be disabled.

    As this function is computationally expensive, it should be run on small datasets. One can 
    use the subset_mode to compute only part of the dataset.

    Building-based features:
    ----------------------
    - buildings_within_buffer
    - total_ft_area_within_buffer
    - av_footprint_area_within_buffer
    - std_footprint_area_within_buffer
    - av_elongation_within_buffer
    - std_elongation_within_buffer
    - av_convexity_within_buffer
    - std_convexity_within_buffer
    - av_orientation_within_buffer
    - std_orientation_within_buffer

    Block-based features:
    -------------------
    - blocks_within_buffer
    - av_block_length_within_buffer
    - std_block_length_within_buffer
    - av_block_footprint_area_within_buffer
    - std_block_footprint_area_within_buffer
    - av_block_av_footprint_area_within_buffer
    - av_block_orientation_within_buffer
    - std_block_orientation_within_buffer
    """

   # Create df with only buildings in the buffer
    df_without_buffer = df[df['in_buffer']==False]

    # If OSM, also remove buildings with no heights
    if osm_mode == True:

            df_without_buffer = df_without_buffer[df_without_buffer['has_height']==True]

   # Create a spatial index
    df_spatial_index = df.sindex

    # make sure to start with the largest buffer
    buffer_sizes.sort(reverse=True)

    if subset_mode == True:

        end = subset

    else:
        end = len(df_without_buffer)


    ## CREATE LISTS/DFs TO STORE RESULTS

    # Create empty result DataFrame
    df_results = pd.DataFrame()

    # List the names of all features
    features_dict = {}

    feature_list = ['buildings_within_buffer','total_ft_area_within_buffer',
    'av_footprint_area_within_buffer','std_footprint_area_within_buffer',
    'av_elongation_within_buffer','std_elongation_within_buffer',
    'av_convexity_within_buffer','std_convexity_within_buffer',
    'av_orientation_within_buffer','std_orientation_within_buffer']

    if block_based==True:

        feature_block_list = ['blocks_within_buffer', 
        'av_block_length_within_buffer', 'std_block_length_within_buffer',
        'av_block_footprint_area_within_buffer', 'std_block_footprint_area_within_buffer',
        'av_block_av_footprint_area_within_buffer',
        'av_block_orientation_within_buffer', 'std_block_orientation_within_buffer']

        feature_list = feature_list + feature_block_list


    ## Create a dictionary to store all the features
    # dict = {buf_size_1: {'ft_1': list, 'ft_2'...}, buf_size_2: {'ft_1': list, 'ft_2'...},..}
    for buf_size in buffer_sizes:

        # first level: buffer size
        features_dict[buf_size] = {}

        # second level: feature name
        for feature in feature_list:

            features_dict[buf_size][feature] = [None] * len(df_without_buffer)


    ## RETRIEVE BUILDINGS IN BUFFER

    for index,row in df_without_buffer.loc[0:end].iterrows():
    
        #print('index')
        print(index)

        # Start with largest buffer size !!
        for buf_index, buf_size in enumerate(buffer_sizes):

            # draw x-meter *circular* buffers (use centroid for circular, real geom for orginal shape)
            buffer = row.geometry.centroid.buffer(buf_size)

            # try:
            
            if buf_index == 0:

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(df_spatial_index.intersection(buffer.bounds))

                # Retrieve possible matches
                possible_matches = df.loc[possible_matches_index] 

            # Identify precise matches that intersect
            precise_matches = possible_matches[possible_matches.intersects(buffer)]

            # except:

            #     print('issue with index')

            #     features_dict[buf_size]['av_footprint_area_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['std_footprint_area_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['av_elongation_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['std_elongation_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['av_convexity_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['std_convexity_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['av_orientation_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['std_orientation_within_buffer'][index] = np.nan
            #     features_dict[buf_size]['total_ft_area_within_buffer'][index] = np.nan
            #     ft_area_around_index = np.nan

            #     continue


            ## COUNT 
            # Remove building i from count
            features_dict[buf_size]['buildings_within_buffer'][index] = len(precise_matches)-1


            ## GEOMETRY-BASED FEATURES

            # if there are other buildings in the buffer
            if len(precise_matches) < 2:

                features_dict[buf_size]['av_footprint_area_within_buffer'][index] = 0
                features_dict[buf_size]['std_footprint_area_within_buffer'][index] = 0
                features_dict[buf_size]['av_elongation_within_buffer'][index] = 0
                features_dict[buf_size]['std_elongation_within_buffer'][index] = 0
                features_dict[buf_size]['av_convexity_within_buffer'][index] = 0
                features_dict[buf_size]['std_convexity_within_buffer'][index] = 0
                features_dict[buf_size]['av_orientation_within_buffer'][index] = 0
                features_dict[buf_size]['std_orientation_within_buffer'][index] = 0
                features_dict[buf_size]['total_ft_area_within_buffer'][index] = 0
                ft_area_around_index = 0 
             

            else:

                # remove the building i
                # try:
                precise_matches = precise_matches.drop(index)

                ## Average footprint area
                features_dict[buf_size]['av_footprint_area_within_buffer'][index] = precise_matches.geometry.area.mean()

                ## Standard deviation footprint area
                features_dict[buf_size]['std_footprint_area_within_buffer'][index] = precise_matches.geometry.area.std()

                ## Average and standard deviation of elongation of buildings within buffer
                elongation = momepy_Elongation(precise_matches).series
                features_dict[buf_size]['av_elongation_within_buffer'][index] = elongation.mean()
                features_dict[buf_size]['std_elongation_within_buffer'][index] = elongation.std()

                ## Average and standard deviation of convexity of buildings within buffer
                convexity = momepy_Convexeity(precise_matches).series
                features_dict[buf_size]['av_convexity_within_buffer'][index] = convexity.mean()
                features_dict[buf_size]['std_convexity_within_buffer'][index] = convexity.std()

                ## Average and standard deviation of orientation of buildings within buffer
                orientation = momepy_Orientation(precise_matches).series
                features_dict[buf_size]['av_orientation_within_buffer'][index] = orientation.mean()
                features_dict[buf_size]['std_orientation_within_buffer'][index] = orientation.std()


                # except:
                # if the building does not fall in a buffer around it's centroid
                print('Building {} must be a castle or something.'.format(index))

                features_dict[buf_size]['av_footprint_area_within_buffer'][index] = 0
                features_dict[buf_size]['std_footprint_area_within_buffer'][index] = 0
                features_dict[buf_size]['av_elongation_within_buffer'][index] = 0
                features_dict[buf_size]['std_elongation_within_buffer'][index] = 0
                features_dict[buf_size]['av_convexity_within_buffer'][index] = 0
                features_dict[buf_size]['std_convexity_within_buffer'][index] = 0
                features_dict[buf_size]['av_orientation_within_buffer'][index] = 0
                features_dict[buf_size]['std_orientation_within_buffer'][index] = 0

                ## Total footprint area within buffer
                # Initialize counter for building i
                ft_area_around_index = []

                # try:

                # Remove the building i
                for j in precise_matches.index:

                    # the building is fully within in the buffer    
                    if precise_matches.geometry[j].within(buffer) == True:

                        # Add the total area
                        ft_area_around_index.append(precise_matches.geometry[j].area)

                    else:
                        # Add only area within the buffer
                        ft_area_around_index.append(precise_matches.geometry[j].intersection(buffer).area)
            
                        # Store 
                        features_dict[buf_size]['total_ft_area_within_buffer'][index] = sum(ft_area_around_index)
          
                # except:                 
                #     print('Building {} seems to have topological problem.'.format(index))

                #     features_dict[buf_size]['total_ft_area_within_buffer'][index] = np.nan


                ## BLOCK-BASED FEATURES

                if block_based==True:

                    ## Blocks in buffer
                    
                    # list of blocks 
                    block_indices_list_around_index = []
                    
                    # list of individual features
                    block_length_around_index = []
                    block_footprint_area_around_index = []
                    block_av_footprint_area_around_index = []
                    block_orientation_around_index = []

                    # iterate through the matches to retrieve the block info
                    for precise_matche_index in precise_matches.index:

                        # if the block (list of indices) is not already in the list of block within buffer                       
                        if df.loc[precise_matche_index]['TouchesIndexes'] not in block_indices_list_around_index:

                            # if the list is longer than 1 (this is a block)
                            if len(df.loc[precise_matche_index]['TouchesIndexes']) > 1:

                                # Append the info from the blocks
                                # list
                                block_indices_list_around_index.append(df.loc[precise_matche_index]['TouchesIndexes'])
                                
                                # list of individual features
                                block_length_around_index.append(df.loc[precise_matche_index]['BlockLength'])
                                block_footprint_area_around_index.append(df.loc[precise_matche_index]['BlockTotalFootprintArea'])
                                block_av_footprint_area_around_index.append(df.loc[precise_matche_index]['AvBlockFootprintArea'])
                                block_orientation_around_index.append(df.loc[precise_matche_index]['BlockOrientation'])


                    # if no blocks found, store 0s
                    if len(block_indices_list_around_index) == 0:

                        features_dict[buf_size]['blocks_within_buffer'][index] = 0
                        features_dict[buf_size]['av_block_length_within_buffer'][index] = 0
                        features_dict[buf_size]['std_block_length_within_buffer'][index] = 0
                        features_dict[buf_size]['av_block_footprint_area_within_buffer'][index] = 0
                        features_dict[buf_size]['std_block_footprint_area_within_buffer'][index] = 0
                        features_dict[buf_size]['av_block_av_footprint_area_within_buffer'][index] = 0
                        features_dict[buf_size]['av_block_orientation_within_buffer'][index] = 0
                        features_dict[buf_size]['std_block_orientation_within_buffer'][index] = 0


                    # else compute statistics
                    else:
                                               
                        ## Number of blocks within buffer
                        features_dict[buf_size]['blocks_within_buffer'][index] = len(block_indices_list_around_index)

                        ## Average block length within buffer
                        features_dict[buf_size]['av_block_length_within_buffer'][index] = pd.Series(block_length_around_index).mean()

                        ## Standard deviation block length within buffer
                        features_dict[buf_size]['std_block_length_within_buffer'][index] = pd.Series(block_length_around_index).std()

                        ## Average block total footprint area within buffer
                        features_dict[buf_size]['av_block_footprint_area_within_buffer'][index] = pd.Series(block_footprint_area_around_index).mean()

                        ## Standard deviation block total footprint area within buffer
                        features_dict[buf_size]['std_block_footprint_area_within_buffer'][index] = pd.Series(block_footprint_area_around_index).std()

                        ## Average average building footprint in blocks within buffer
                        features_dict[buf_size]['av_block_av_footprint_area_within_buffer'][index] = pd.Series(block_av_footprint_area_around_index).mean()

                        ## Average block orientation within buffer
                        features_dict[buf_size]['av_block_orientation_within_buffer'][index] = pd.Series(block_orientation_around_index).mean()

                        ## Standard deviation block orientation within buffer
                        features_dict[buf_size]['std_block_orientation_within_buffer'][index] = pd.Series(block_orientation_around_index).std()

        
    for buf_size in buffer_sizes:

        for feature in feature_list:

            df_results.insert(loc=0, column='{}_{}'.format(feature,buf_size), value=features_dict[buf_size][feature]) 

    df_results = df_results.fillna(0)

    return(df_results)




def features_streets(df,
    df_streets,
    df_intersections,
    buffer_sizes=None,
    subset_mode = False,
    subset = None,

    ):
    """
    Returns a DataFrame with features about the streets and intersections surrounding 
    each building within given distances (circular buffers).

    Features closest intersection:
    ----------------------------
    - distance_to_closest_intersection

    Features closest street:
    --------------
    - distance_to_closest_road
    - street_length_closest_road
    - street_width_av_closest_road
    - street_width_std_closest_road
    - street_openness_closest_road
    - (street_linearity_closest_road)

    - street_closeness_global_closest_road
    - street_betweeness_global_closest_road
    - street_betweeness_500_closest_road

    Features intersections buffer:
    ----------------------
    - intersection_count_within_buffer 

    Features streets buffer:
    -----------------------
    - street_length_total_within_buffer
    - street_length_av_within_buffer
    - street_length_std_within_buffer
    - street_length_total_inter_buffer

    - (street_orientation_std_inter_buffer)
    - (street_linearity_av_inter_buffer)
    - (street_linearity_std_inter_buffer)
    - street_width_av_inter_buffer
    - street_width_std_inter_buffer

    - street_betweeness_global_max_inter_buffer
    - street_betweeness_global_av_inter_buffer
    - street_betweeness_500_max_inter_buffer
    - street_betweeness_500_av_inter_buffer
    """

    # Create empty result DataFrame
    df_results = pd.DataFrame()


    # make sure to start with the largest buffer
    buffer_sizes.sort(reverse=True)

    # subset mode
    if subset_mode == True:

        end = subset

    else:
        end = len(df)

    # create spatial indexes
    str_spatial_index = df_streets.sindex
    int_spatial_index = df_intersections.sindex

    ## CREATE DICT TO STORE RESULTS 

    features_dict_closest = {}

    features_dict_buffer = {}

    list_closest =  ['distance_to_closest_intersection','distance_to_closest_road','street_length_closest_road',   
        'street_width_av_closest_road','street_width_std_closest_road','street_openness_closest_road',
        'street_closeness_global_closest_road','street_betweeness_global_closest_road','street_closeness_500_closest_road']

    list_within_buffer = ['intersection_count_within_buffer','street_length_total_within_buffer','street_length_av_within_buffer',
        'street_length_std_within_buffer','street_length_total_inter_buffer','street_width_av_inter_buffer',
        'street_width_std_inter_buffer','street_betweeness_global_max_inter_buffer','street_betweeness_global_av_inter_buffer',
        'street_closeness_500_max_inter_buffer','street_closeness_500_av_inter_buffer']

    ## Create a dictionary to store all the closest-object features
    for feature in list_closest:

        features_dict_closest[feature] = [None] * (len(df)+1)

    print('Len distance_to_closest_intersection:')
    print(len(features_dict_closest['distance_to_closest_intersection']))


    ## Create a dictionary to store all the within-buffer features
    # dict = {buf_size_1: {'ft_1': list, 'ft_2'...}, buf_size_2: {'ft_1': list, 'ft_2'...},..}
    for buf_size in buffer_sizes:

        # first level: buffer size
        features_dict_buffer[buf_size] = {}

        # second level: feature name
        for feature in list_within_buffer:

            features_dict_buffer[buf_size][feature] = [None] * (len(df)+1)

    ################
    ### FEATURES ###
    ################


    for index,row in df.loc[0:end].iterrows():

        print(index)

        #### CLOSEST OBJECTS #####

        ### CLOSEST INTERSECTION

        ## retrieve closest intersection to the building

        # empty list of matches indexes
        possible_matches_index = []

        # start with small buffer
        buffer_size = 100

        # get a buffer around the building
        buffer = row.geometry.centroid.buffer(buffer_size)

        # retrieve matched index
        possible_matches_index = list(int_spatial_index.intersection(buffer.bounds))

        # if no index was retrieve
        if possible_matches_index == []:

            # until one index gets retrieved
            while possible_matches_index == []:

                # increase buffer size
                buffer_size += 100

                # draw
                buffer = row.geometry.buffer(buffer_size)

                # retrieve indexes
                possible_matches_index = list(int_spatial_index.intersection(buffer.bounds))

        # retrieve rows
        precise_matches = df_intersections.loc[possible_matches_index]

        # for all matches
        for index_closest,row_closest in precise_matches.iterrows():
            
            # add distance to the building
            precise_matches.loc[index_closest,'distance'] = df.loc[index,'geometry'].distance(precise_matches.loc[index_closest,'geometry'])

        # sort by distance
        precise_matches = precise_matches.sort_values(by=['distance'])

        # retrieve closest
        closest_int = precise_matches.iloc[0]

        ## retrieve features

        # try:
        features_dict_closest['distance_to_closest_intersection'][index] = closest_int.distance
        # except:
        #     print(len(features_dict_closest['distance_to_closest_intersection']))

        ### CLOSEST STREET 


        ## retrieve closest street to the building

        # empty list of matches indexes
        possible_matches_index = []

        # start with small buffer
        buffer_size = 100

        # get a buffer around the building
        buffer = row.geometry.centroid.buffer(buffer_size)

        # retrieve matched index
        possible_matches_index = list(str_spatial_index.intersection(buffer.bounds))

        # if no index was retrieve
        if possible_matches_index == []:

            # until one index gets retrieved
            while possible_matches_index == []:

                # increase buffer size
                buffer_size += 100

                # draw
                buffer = row.geometry.buffer(buffer_size)

                # retrieve indexes
                possible_matches_index = list(str_spatial_index.intersection(buffer.bounds))

        # retrieve rows
        precise_matches = df_streets.loc[possible_matches_index]

        # for all matches
        for index_closest,row_closest in precise_matches.iterrows():
            
            # add distance to the building
            precise_matches.loc[index_closest,'distance'] = df.loc[index,'geometry'].distance(precise_matches.loc[index_closest,'geometry'])

        # sort by distance
        precise_matches = precise_matches.sort_values(by=['distance'])

        # retrieve closest
        closest_str = precise_matches.iloc[0]

        ## retrieve features

        features_dict_closest['distance_to_closest_road'][index] = closest_str.distance
        features_dict_closest['street_length_closest_road'][index] = closest_str.length
        features_dict_closest['street_width_av_closest_road'][index] = closest_str.widths
        features_dict_closest['street_width_std_closest_road'][index] = closest_str.width_deviation
        features_dict_closest['street_openness_closest_road'][index] = closest_str.openness
        features_dict_closest['street_betweeness_global_closest_road'][index] = closest_str.betweenness_metric_e
        features_dict_closest['street_closeness_global_closest_road'][index] = closest_str.closeness_global
        features_dict_closest['street_closeness_500_closest_road'][index] = closest_str.closeness500


        #### WITHIN-BUFFER FEATURES ####


        ## RETURN OBJECTS WITHIN BUFFERS

        # Start with largest buffer size 
        for buf_index, buf_size in enumerate(buffer_sizes):

            # draw x-meter *circular* buffers (use centroid for circular, real geom for orginal shape)
            buffer = row.geometry.centroid.buffer(buf_size)

            if buf_index == 0:

                ## Possible intersections

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(int_spatial_index.intersection(buffer.bounds))
                # Retrieve possible matches
                possible_matches_int = df_intersections.loc[possible_matches_index] 

                ## Possible streets

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(str_spatial_index.intersection(buffer.bounds))
                # Retrieve possible matches
                possible_matches_str = df_streets.loc[possible_matches_index] 


            ## Identify precise matches that intersect
            precise_matches_int = possible_matches_int[possible_matches_int.intersects(buffer)]

            precise_matches_str_inter = possible_matches_str[possible_matches_str.intersects(buffer)]
            precise_matches_str_within = possible_matches_str[possible_matches_str.within(buffer)]


            ## Compute features

            features_dict_buffer[buf_size]['intersection_count_within_buffer'][index] = len(precise_matches_int)

            features_dict_buffer[buf_size]['street_length_total_within_buffer'][index] = precise_matches_str_within.length.sum()
            features_dict_buffer[buf_size]['street_length_av_within_buffer'][index] = precise_matches_str_within.length.mean()
            features_dict_buffer[buf_size]['street_length_std_within_buffer'][index] = precise_matches_str_within.length.std()

            features_dict_buffer[buf_size]['street_length_total_inter_buffer'][index] = precise_matches_str_inter.length.sum()

            features_dict_buffer[buf_size]['street_width_av_inter_buffer'][index] = precise_matches_str_inter.widths.mean()
            features_dict_buffer[buf_size]['street_width_std_inter_buffer'][index] = precise_matches_str_inter.widths.std()

            features_dict_buffer[buf_size]['street_betweeness_global_max_inter_buffer'][index] = precise_matches_str_inter.betweenness_metric_e.max()
            features_dict_buffer[buf_size]['street_betweeness_global_av_inter_buffer'][index] = precise_matches_str_inter.betweenness_metric_e.mean()
            features_dict_buffer[buf_size]['street_closeness_500_max_inter_buffer'][index] = precise_matches_str_inter.closeness500.max()
            features_dict_buffer[buf_size]['street_closeness_500_av_inter_buffer'][index] = precise_matches_str_inter.closeness500.mean()

    # store within-buffer features in df
    for buf_size in buffer_sizes:

            for feature in list_within_buffer:

                df_results.insert(loc=0, column='{}_{}'.format(feature,buf_size), value=features_dict_buffer[buf_size][feature]) 

    # store closest features in df
    for feature in list_closest:

                df_results.insert(loc=0, column='{}'.format(feature), value=features_dict_closest[feature]) 

    df_results = df_results.fillna(0)

    return(df_results)



def features_streets_based_block(df_buildings,
    df_streets_based_block,
    buffer_sizes=None,
    subset_mode = False,
    subset = None,
    osm_mode = False
    ):
    """
    Returns a DataFrame with features about the street-based block where the building
    falls and the blocks intersecting buffers.

    Feature own block:
    ----------------
    - street_based_block_area
    - street_based_block_phi
    - street_based_block_corners

    Features intersection with buffer:
    ---------------------
    - street_based_block_number_inter_buffer
    - street_based_block_av_area_inter_buffer
    - street_based_block_std_area_inter_buffer
    - street_based_block_av_phi_inter_buffer
    - street_based_block_std_phi_inter_buffer
    - street_based_block_std_orientation_inter_buffer
    """

    # Create empty result DataFrame
    df_results = pd.DataFrame()

   # Create df with only buildings in the buffer
    df_without_buffer = df_buildings[df_buildings['in_buffer']==False]

    # If OSM, also remove buildings with no heights
    if osm_mode == True:

            df_without_buffer = df_without_buffer[df_without_buffer['has_height']==True]

    # make sure to start with the largest buffer
    buffer_sizes.sort(reverse=True)

    # subset mode
    if subset_mode == True:

        end = subset

    else:
        end = len(df_without_buffer)

    # create spatial indexes
    block_spatial_index = df_streets_based_block.sindex

    ## CREATE DICT TO STORE RESULTS 

    dict_own_block = {}

    dict_inter_buffer = {}

    list_own_block = ['street_based_block_area','street_based_block_phi','street_based_block_corners']
    
    list_inter_buffer = ['street_based_block_number_inter_buffer','street_based_block_av_area_inter_buffer','street_based_block_std_area_inter_buffer',
    'street_based_block_av_phi_inter_buffer','street_based_block_std_phi_inter_buffer','street_based_block_std_orientation_inter_buffer']


    ## Create a dictionary to store all the closest-object features
    for feature in list_own_block:

        dict_own_block[feature] = [None] * (len(df_without_buffer)+1)

    ## Create a dictionary to store all the within-buffer features
    # dict = {buf_size_1: {'ft_1': list, 'ft_2'...}, buf_size_2: {'ft_1': list, 'ft_2'...},..}
    for buf_size in buffer_sizes:

        # first level: buffer size
        dict_inter_buffer[buf_size] = {}

        # second level: feature name
        for feature in list_inter_buffer:

            dict_inter_buffer[buf_size][feature] = [None] * (len(df_without_buffer)+1)

    ################
    ### FEATURES ###
    ################

    for index,row in df_without_buffer.loc[0:end].iterrows():

        print(index)

    ## OWN BLOCK

        try:

            possible_matches_index = list(block_spatial_index.intersection(row.geometry.bounds))
            possible_matches = df_streets_based_block.loc[possible_matches_index] 
            precise_match = possible_matches[possible_matches.contains(row.geometry)]

            if len(precise_match)>1:

                print('Multiple blockkks!')

            else:
                dict_own_block['street_based_block_area'][index] = precise_match.iloc[0].area
                dict_own_block['street_based_block_phi'][index] = precise_match.iloc[0].Phi
                dict_own_block['street_based_block_corners'][index] = precise_match.iloc[0].Corners

        except:
            print('Building {} does not fall within an existing block.'.format(index))

            dict_own_block['street_based_block_area'][index] = 0
            dict_own_block['street_based_block_phi'][index] = 0
            dict_own_block['street_based_block_corners'][index] = 0 

    ## FEATURES WITHIN BUFFER

        # Start with largest buffer size 
        for buf_index, buf_size in enumerate(buffer_sizes):

            # draw x-meter *circular* buffers (use centroid for circular, real geom for orginal shape)
            buffer = row.geometry.centroid.buffer(buf_size)

            if buf_index == 0:

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(block_spatial_index.intersection(buffer.bounds))
                # Retrieve possible matches
                possible_matches = df_streets_based_block.loc[possible_matches_index] 

            ## Identify precise matches that intersect
            precise_matches = possible_matches[possible_matches.intersects(buffer)]

            ## Compute features
            dict_inter_buffer[buf_size]['street_based_block_number_inter_buffer'][index] = len(precise_matches)
            dict_inter_buffer[buf_size]['street_based_block_av_area_inter_buffer'][index] = precise_matches.area.mean()
            dict_inter_buffer[buf_size]['street_based_block_std_area_inter_buffer'][index] = precise_matches.area.std()
            dict_inter_buffer[buf_size]['street_based_block_av_phi_inter_buffer'][index] = precise_matches.Phi.mean()
            dict_inter_buffer[buf_size]['street_based_block_std_phi_inter_buffer'][index] = precise_matches.Phi.std()
            dict_inter_buffer[buf_size]['street_based_block_std_orientation_inter_buffer'][index] = precise_matches.streets_based_block_orientation.std()


    # store within-buffer features in df
    for buf_size in buffer_sizes:

            for feature in list_inter_buffer:

                df_results.insert(loc=0, column='{}_{}'.format(feature,buf_size), value=dict_inter_buffer[buf_size][feature]) 

    # store closest features in df
    for feature in list_own_block:

                df_results.insert(loc=0, column='{}'.format(feature), value=dict_own_block[feature]) 

    df_results = df_results.fillna(0)


    return(df_results)



def features_urban_atlas(df_buildings,
    df_urban_atlas,
    id_col=None,
    buffer_sizes=None,
    subset_mode = False,
    subset = None,
    osm_mode = False
    ):
    """
    Returns a DataFrame with features about the land use classes surrounding 
    each building within given distances (circular buffers).

    Feature land use building:
    ------------------------
    - building_in_lu_agricultural
    - building_in_lu_industrial_commercial
    - building_in_lu_natural_semi_natural
    - building_in_lu_railways
    - building_in_lu_urban_fabric
    - building_in_lu_urban_green
    - building_in_lu_wastelands

    Feature land use area within buffer:
    ---------------------------------
    - lu_agricultural_within_buffer
    - lu_industrial_commercial_within_buffer
    - lu_natural_semi_natural_within_buffer
    - lu_railways_within_buffer
    - lu_roads_within_buffer
    - lu_urban_fabric_within_buffer
    - lu_urban_green_within_buffer
    - lu_wastelands_within_buffer
    - lu_water_within_buffer
    """

   # Create df with only buildings in the buffer
    df_without_buffer = df_buildings[df_buildings['in_buffer']==False]

    # keep only in UA
    df_without_buffer = df_buildings[df_buildings['in_UA']==True]

    # remove also those buildings that are not within 500m of the UA adjusted boundary
    df_without_buffer = df_without_buffer[df_without_buffer['in_UA_500_buffer']==False]

    # If OSM, also remove buildings with no heights
    if osm_mode == True:

            df_without_buffer = df_without_buffer[df_without_buffer['has_height']==True]

    # make sure to start with the largest buffer
    buffer_sizes.sort(reverse=True)

    # subset mode
    if subset_mode == True:

        end = subset

    else:
        end = len(df_without_buffer)-1

    # create spatial index for the land use classes
    ua_spatial_index = df_urban_atlas.sindex

    # Create empty result DataFrame
    df_results = pd.DataFrame()


    ## CREATE DICT TO STORE RESULTS

    # building_in ft dict
    building_in_fts = {'building_in_lu_agricultural': [None] * len(df_without_buffer),
    'building_in_lu_industrial_commercial': [None] * len(df_without_buffer),
    'building_in_lu_natural_semi_natural': [None] * len(df_without_buffer),
    'building_in_lu_railways': [None] * len(df_without_buffer),
    'building_in_lu_urban_fabric': [None] * len(df_without_buffer),
    'building_in_lu_urban_green': [None] * len(df_without_buffer),
    'building_in_lu_wastelands': [None] * len(df_without_buffer)}

    # mapping ft to lu class for one-hot encoding (building in)
    dict_ua_to_ft = {'building_in_lu_agricultural': 'Agricultural',
             'building_in_lu_industrial_commercial': 'Industrial, commercial area',
             'building_in_lu_natural_semi_natural': 'Natural and semi-natural',
             'building_in_lu_railways': 'Railways',
             'building_in_lu_urban_fabric': 'Urban fabric',
             'building_in_lu_urban_green': 'Urban green',
             'building_in_lu_wastelands': 'Wastelands and co'
             }

    # mapping lu class to ft (within buffer)
    lu_classes_mapping = {'Agricultural': 'lu_agricultural_within_buffer',
        'Industrial, commercial area':'lu_industrial_commercial_within_buffer',
        'Natural and semi-natural':'lu_natural_semi_natural_within_buffer',
        'Railways':'lu_railways_within_buffer',
        'Roads':'lu_roads_within_buffer',
        'Urban fabric':'lu_urban_fabric_within_buffer',
        'Urban green':'lu_urban_green_within_buffer',
        'Wastelands and co':'lu_wastelands_within_buffer',
        'Water':'lu_water_within_buffer'
        }

    # list fts (within buffer)
    lu_within_buffer_fts_list = ['lu_agricultural_within_buffer',
        'lu_industrial_commercial_within_buffer',
        'lu_natural_semi_natural_within_buffer',
        'lu_railways_within_buffer',
        'lu_roads_within_buffer',
        'lu_urban_fabric_within_buffer',
        'lu_urban_green_within_buffer',
        'lu_wastelands_within_buffer',
        'lu_water_within_buffer']


    # lu_within buffer ft dict
    lu_within_buffer_fts = {}

    # create levels
    for buf_size in buffer_sizes:

        # first level: buffer size
        lu_within_buffer_fts[buf_size] = {}

        # second level: feature name
        for feature in lu_within_buffer_fts_list:

            lu_within_buffer_fts[buf_size][feature] = [None] * len(df_without_buffer)


    ## RETRIEVE LAND USE OF BUILDING 

    print('Retrieve within which land use the building is located...')

    for index,row in df_without_buffer.loc[0:end].iterrows():

        print(index)

        try:
    
            # retrieve intersection building lu polygons
            possible_matches_index = list(ua_spatial_index.intersection(row.geometry.bounds))
            possible_matches = df_urban_atlas.loc[possible_matches_index] 
            precise_matches = possible_matches[possible_matches.intersects(row.geometry)]

            # compute intersection area between building and polygons
            precise_matches.loc[:,'intersection'] = precise_matches.geometry.intersection(row.geometry).area

            # retrieve the largest intersection
            precise_matches = precise_matches.loc[precise_matches['intersection'].idxmax()]

            # one-hot encoding
            for ft in dict_ua_to_ft.keys():

                # if the the building is within a given class give 1
                if precise_matches['ft_typology'] == dict_ua_to_ft[ft]:

                    building_in_fts[ft][index] = 1

                # and give 0 to all the others
                else:

                    building_in_fts[ft][index] = 0

        except:

            print('{} in no land use poly?'.format(index))

            # one-hot encoding
            for ft in dict_ua_to_ft.keys():

                    building_in_fts[ft][index] = 0


    ## RETRIEVE LAND USES IN BUFFER

    print('Retrieve land uses in buffer...')

    for index,row in df_without_buffer.loc[0:end].iterrows():

        print(index)

        # Start with largest buffer size !!
        for buf_index, buf_size in enumerate(buffer_sizes):

            ## retrieve lu polygons that intersect the buffer

            # draw x-meter *circular* buffers (use centroid for circular, real geom for orginal shape)
            buffer = row.geometry.centroid.buffer(buf_size)

            if buf_index == 0:

                # Intersection spatial index and bounding box of the building i
                possible_matches_index = list(ua_spatial_index.intersection(buffer.bounds))

                # Retrieve possible matches
                possible_matches = df_urban_atlas.loc[possible_matches_index] 

            # Identify precise matches that intersect
            precise_matches = possible_matches[possible_matches.intersects(buffer)]


            ## retrieve the intersection polygons
            for index2,_ in precise_matches.iterrows():
                
                # retrieve the polygon and write in wkt (otherwise empty polygons generate errors)
                precise_matches.loc[index2,'intersection'] = precise_matches.loc[index2,'geometry'].intersection(buffer).wkt

            # remove empty polygons (which have not matched)
            precise_matches = precise_matches[precise_matches['intersection'] != 'POLYGON EMPTY' ]

            # load geometries from wkt
            precise_matches.loc[:,'intersection'] = precise_matches['intersection'].apply(wkt.loads)

            # change geometry column
            precise_matches = precise_matches.set_geometry('intersection')

            ## compute areas

            # add area to each polygon
            precise_matches['area'] = precise_matches['intersection'].area

            # sum by class in serie
            areas = precise_matches.groupby('ft_typology')['area'].sum()

            # check which classes are not in the buffer
            add_index = list(set(lu_classes_mapping.keys()).difference(areas.index))

            # these will all get 0 as value
            add = pd.Series([0] * len(add_index), index = add_index)

            # add them to serie
            areas = areas.append(add)

            ## save into lists 
            for lu in lu_classes_mapping.keys():

                lu_within_buffer_fts[buf_size][lu_classes_mapping[lu]][index] = areas[lu]


    # add buffer based features
    for buf_size in buffer_sizes:

        for ft in lu_within_buffer_fts[buf_size].keys():

            df_results.insert(loc=0, column='{}_{}'.format(ft,buf_size), value=lu_within_buffer_fts[buf_size][ft]) 

    # add building in features
    for ft in building_in_fts.keys():

        df_results.insert(loc=0, column='{}'.format(ft), value=building_in_fts[ft]) 


    id = df_without_buffer.loc[0:end][id_col]    
    df_results.insert(loc=0, column='id', value=id) 

    df_results = df_results.fillna(0)

    return(df_results)


def features_city_level(buildings_for_fts,
    geom_boundary,
    df_buildings_w_building_features,
    df_intersections,
    df_streets,
    df_streets_based_block,
    df_urban_atlas = None,
    osm_mode = False,
    ua_available = False,
    ):
    """
    Returns a DataFrame with features at the city level.

    Important - the building df must contain already features for blocks, as they are needed in the computation
    
    Features boundaries:
    ------------------
    - area_city
    - phi_city

    Features buildings:
    ------------------
    - total_buildings_city
    - av_building_footprint_city
    - std_building_footprint_city
    - num_detached_buildings
    - block_2_to_5
    - block_6_to_10
    - block_11_to_20
    - block_20+

    Features intersections:
    ----------------------
    - total_intersection_city

    Features streets:
    ----------------
    - total_length_street_city
    - av_length_street_city

    Features blocks:
    ---------------
    - total_number_block_city
    - av_area_block_city
    - std_area_block_city

    Features Urban Atlas:
    --------------------
    - lu_agricultural_within_buffer
    - lu_industrial_commercial_within_buffer
    - lu_natural_semi_natural_within_buffer
    - lu_railways_within_buffer
    - lu_roads_within_buffer
    - lu_urban_fabric_within_buffer
    - lu_urban_green_within_buffer
    - lu_wastelands_within_buffer
    - lu_water_within_buffer

    """

   # Create df with only buildings in the buffer
    df_without_buffer = buildings_for_fts[buildings_for_fts['in_buffer']==False]

    # If OSM, also remove buildings with no heights
    if osm_mode == True:

            df_without_buffer = df_without_buffer[df_without_buffer['has_height']==True]

    # Create empty result DataFrame
    df_results = pd.DataFrame()


    ### Features boundaries ###

    ## Area city
    area_city = geom_boundary.area
    df_results['area_city'] = [area_city] * len(df_without_buffer)

    ## Phi city

    if type(geom_boundary) == MultiPolygon:

        geom_boundary = max(geom_boundary, key=lambda a: a.area)

    # Draw exterior ring of the building
    exterior = geom_boundary.exterior

    # Compute max distance to a point
    max_dist = geom_boundary.centroid.hausdorff_distance(exterior)

    # Draw smallest circle around building
    min_circle = geom_boundary.centroid.buffer(max_dist)

    # Area of the circle
    circle_area = min_circle.area

    # Compute phi
    phi_city = geom_boundary.area/circle_area

    df_results['phi_city'] = [phi_city] * len(df_without_buffer)


    ### Features buildings ###

    ## total_buildings_city
    total_buildings_city = len(df_buildings_w_building_features)
    df_results['total_buildings_city'] = [total_buildings_city] * len(df_without_buffer)

    ## av_building_footprint_city
    av_building_footprint_city = df_buildings_w_building_features.geometry.area.mean()
    df_results['av_building_footprint_city'] = [av_building_footprint_city] * len(df_without_buffer)

    ## std_building_footprint_city
    std_building_footprint_city = df_buildings_w_building_features.geometry.area.std()
    df_results['std_building_footprint_city'] = [std_building_footprint_city] * len(df_without_buffer)

    ## blocks
    single_blocks = df_buildings_w_building_features.drop_duplicates(subset = 'TouchesIndexes')
    counts = dict(Counter(single_blocks.BlockLength))

    new_counts = {'1':0,'2-5':0,'6-10':0,'11-20':0,'20+':0}
    for key in counts.keys():
        if key == 1:
            new_counts['1'] += counts[key]
        if key in range(2,5):
            new_counts['2-5'] += counts[key]
        if key in range(6,10):
            new_counts['6-10'] += counts[key]
        if key in range(11,20):
            new_counts['11-20'] += counts[key]
        if key > 20:
            new_counts['20+'] += counts[key]

    df_results['num_detached_buildings'] = new_counts['1']
    df_results['block_2_to_5'] = new_counts['2-5']
    df_results['block_6_to_10'] = new_counts['6-10']
    df_results['block_11_to_20'] = new_counts['11-20']
    df_results['block_20+'] = new_counts['20+']


    ### Features intersections ###

    #total_intersection_city
    total_intersection_city = len(df_intersections)
    df_results['total_intersection_city'] = [total_intersection_city] * len(df_without_buffer)


    ### Features streets ### 

    # total_length_street_city
    total_length_street_city = df_streets.geometry.length.sum()
    df_results['total_length_street_city'] = [total_length_street_city] * len(df_without_buffer)

    # av_length_street_city
    av_length_street_city = df_streets.geometry.length.mean()
    df_results['av_length_street_city'] = [av_length_street_city] * len(df_without_buffer)

    ### Features street-based blocks ###

    # total_number_block_city
    total_number_block_city = len(df_streets_based_block)
    df_results['total_number_block_city'] = [total_number_block_city] * len(df_without_buffer)

    # av_area_block_city
    av_area_block_city = df_streets_based_block.geometry.area.mean()
    df_results['av_area_block_city'] = [av_area_block_city] * len(df_without_buffer)

    # std_area_block_city
    std_area_block_city = df_streets_based_block.geometry.area.std()
    df_results['std_area_block_city'] = [std_area_block_city] * len(df_without_buffer)


    ### Features urban atlas ###
    if ua_available == True:

        # lu_agricultural_within_buffer
        lu_agricultural_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Agricultural'].geometry.area.sum()
        df_results['lu_agricultural_within_buffer'] = [lu_agricultural_within_buffer] * len(df_without_buffer)

        # lu_industrial_commercial_within_buffer
        lu_industrial_commercial_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Industrial, commercial area'].geometry.area.sum()
        df_results['lu_industrial_commercial_within_buffer'] = [lu_industrial_commercial_within_buffer] * len(df_without_buffer)

        # lu_natural_semi_natural_within_buffer
        lu_natural_semi_natural_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Natural and semi-natural'].geometry.area.sum()
        df_results['lu_natural_semi_natural_within_buffer'] = [lu_natural_semi_natural_within_buffer] * len(df_without_buffer)

        # lu_railways_within_buffer
        lu_railways_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Railways'].geometry.area.sum()
        df_results['lu_railways_within_buffer'] = [lu_railways_within_buffer] * len(df_without_buffer)

        # lu_roads_within_buffer
        lu_roads_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Roads'].geometry.area.sum()
        df_results['lu_roads_within_buffer'] = [lu_roads_within_buffer] * len(df_without_buffer)

        # lu_urban_fabric_within_buffer
        lu_urban_fabric_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Urban fabric'].geometry.area.sum()
        df_results['lu_urban_fabric_within_buffer'] = [lu_urban_fabric_within_buffer] * len(df_without_buffer)

        # lu_urban_green_within_buffer
        lu_urban_green_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Urban green'].geometry.area.sum()
        df_results['lu_urban_green_within_buffer'] = [lu_urban_green_within_buffer] * len(df_without_buffer)

        # lu_wastelands_within_buffer
        lu_wastelands_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Wastelands and co'].geometry.area.sum()
        df_results['lu_wastelands_within_buffer'] = [lu_wastelands_within_buffer] * len(df_without_buffer)

        # lu_water_within_buffer
        lu_water_within_buffer = df_urban_atlas[df_urban_atlas['ft_typology'] == 'Water'].geometry.area.sum()
        df_results['lu_water_within_buffer'] = [lu_water_within_buffer] * len(df_without_buffer)

    return(df_results)





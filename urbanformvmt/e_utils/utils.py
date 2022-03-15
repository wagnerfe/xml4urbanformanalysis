from shapely.geometry import Point, MultiPoint, Polygon
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def make_pizza(df, do_plot=False, share=0.2, angle=2*np.pi/2):
    """
    cuts a pizza slice out of df based on geometry
    If row is element of slice the entry in columns "pizza" is "test"
    else train. 
    In:
        df: geopandas.GeoDataFrame; needs geometry column with one shapely.geometry.Point per row
        do_plot: bool; plots if True
        share: float; share of examples used for testing
        angle: float; angle of center of slice
    Out:
        updated GeoDataFrame
    """
    
    center = MultiPoint(df.geometry).centroid

    def assign(point, center, angle):
    
        north = np.array([np.cos(angle), np.sin(angle)])
        
        vector = np.array([point.x - center.x, point.y - center.y])
        angle = np.arccos(np.inner(vector, north) / np.linalg.norm(vector))
    
        return angle
        
    
    df["pizza"] = df.geometry.apply(lambda entry: assign(entry, center, angle))
    
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n+1) / n
        return(x,y)
    
    x, y = ecdf(df["pizza"])
    
    threshold = len(y[y < share])
    threshold = x[threshold]
    
    def assign_task(entry, threshold):
        if entry < threshold:
            return "test"
        else:
            return "train"
        
    df["pizza"] = df["pizza"].apply(lambda entry: assign_task(entry, threshold))
    # to avoid returning the same df as input!
    df_out = df.copy()

    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
        df_out.plot(ax=ax, column="pizza")
        plt.show()
        
    return df_out



def fit_dist(df, column="tripdistancemeters", dist="gamma", do_plot=True, savefig=None):
    """
    Fits  distribution to data
    Args:
        df: (Geo)DataFrame; data with column of interest
        column: str; column of interest
        dist: str; name of ditribution used for the fit
        do_plot: bool; plots the distribution if True
        savefig: None or str; if str stores resulting plot as savefig
    Out:
        params: list; parameters of fitted distribution
    """

    df = df[column]
    
    dist = getattr(stats, dist)
    params = dist.fit(a)
    
    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.hist(df, bins=np.linspace(min(df), max(df), steps), edgecolor="k", linewidth=0.25, density=True)
        
        x = np.linspace(min(df), max(df), 1000)
        params = gamma.fit(a)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        pdf = gamma.pdf(x, loc=loc, scale=scale, *arg)
        ax.plot(x, pdf, 'k-', lw=2, label='frozen pdf')
        
        if savefig is not None:
            plt.savefig(savefig)
        
        plt.show()
    
    return params

def delete_hex_on_boundary(gdf, gdf_city_bound, crs):
    """
    Function to take only hexagons that are located within the berlin boundary.
    Args:
        gdf = mean trips per hex
        gdf_city_bound = boundary of city 
        crs = crs of gdf and gdf_city_bound
    Out:
        gdf_out = gdf with only hex that are inside of city bound

    Author: Felix
    Date: 10.08.21 
    """
    # take hex column
    gdf_hex = gdf.drop(columns='geometry')
    gdf_hex = gdf_hex.rename(columns={'h3_hexagon':'geometry'})
    gdf_hex = gpd.GeoDataFrame(gdf_hex,crs=crs)
    
    # sjoin of berlin bound and hex geometries
    gdf_hex_in = gpd.sjoin(gdf_city_bound,gdf_hex,op = 'contains', how = 'inner')
    gdf_out = gdf.iloc[gdf_hex_in.index_right].reset_index(drop=True)
    return gdf_out


def pizza_oppo_split(gdf,num_folds=5, pos=0):
    """
    Function to calculate Oppo Split.
    Args:
    gdf: input gdf with point geometrys
    num_folds: number of folds 
    pos: position of fold

    Returns: 
    gdf_train: traing dataset (80% of gdf)
    gdf_test: test dataset (20% of gdf)
    """
    # we utilise existing pizza function but take piiza slice + oppo slice to reduce high variance in folds
    df_pizza = make_pizza(gdf,do_plot=False,share = 1/(2*num_folds),angle=pos*2*np.pi/(2*num_folds)).copy()
    df_pizza_oppo = make_pizza(gdf,do_plot=False,share = 1/(2*num_folds),angle=(pos+num_folds)*2*np.pi/(2*num_folds)).copy()  

    # assign all rows that are either test in pizza or oppo pizza to gdf_test
    gdf_test = gdf[(df_pizza.pizza=='test') | (df_pizza_oppo.pizza=='test')]
    # assign all remaining rows to gdf_train
    gdf_train = gdf[(df_pizza.pizza=='train') & (df_pizza_oppo.pizza=='train')]
    return gdf_train, gdf_test  
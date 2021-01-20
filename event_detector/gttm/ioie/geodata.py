# -*- coding: UTF8 -*-
from datetime import datetime

import numpy as np
from pathlib import Path
# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from shapely.geometry import Point
from geopandas import GeoDataFrame
import pandas as pd
import os, sys
import shapely
from matplotlib.collections import PolyCollection

os.environ["PROJ_LIB"] = os.path.dirname(sys.executable) + os.sep + "Library\\share"

# default_srid = 3006
default_srid = 3857  # WGS_1984_Web_Mercator_Auxiliary_Sphere
default_crs = {'init': 'epsg:{}'.format(default_srid)}


def get_default_srid():
    return default_srid


def get_default_crs():
    return default_crs


def add_geometry(df: pd.DataFrame, study_area: shapely.geometry.box = None, crs=default_crs, verbose=False):
    if verbose:
        print('\tStart adding geometry...')
    s_time = datetime.now()

    if df.shape[0] <= 0:
        return None
    df['geometry'] = list(zip(df.x, df.y))
    df['geometry'] = df['geometry'].apply(Point)
    base_crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(df, crs=base_crs, geometry='geometry').copy()

    if study_area is not None:
        gdf = gdf[gdf.within(study_area)]

    if crs is not None:
        gdf = project(gdf, crs=crs)

    # if save_map:
    #    generate_2d_map(gdf=gdf, show_map=show_map, save_map=save_map, map_title=map_title, map_filepath=map_filepath)
    dur = datetime.now() - s_time
    if verbose:
        print('\tAdding geometry was finished ({} seconds).'.format(dur.seconds))
    return gdf


def project(gdf: GeoDataFrame, crs=None):
    if crs is not None:
        gdf = gdf.to_crs(crs=crs)
    return gdf


# def show_wms():
#     map = Basemap(llcrnrlon=8.35, llcrnrlat=41.225, urcrnrlon=10.01, urcrnrlat=43.108,
#                   projection='cyl', epsg=4326)

#     wms_server = "http://www.ga.gov.au/gis/services/topography/Australian_Topography/MapServer/WMSServer"
#     wms_server = "http://wms.geosignal.fr/metropole?"

#     map.wmsimage(wms_server, layers=[
#         "Communes", "Nationales", "Regions"], verbose=True)
#     plt.show()

import numpy as np
import pandas as pd

EARTH_RADIUS = 6371

def calc_dist(df_in: pd.DataFrame, df_out: pd.DataFrame ) -> float:
    """
    Calculate the haversine and Manhattan distances between two
    points on the earth (specified in decimal degrees)
    Vectorized version for pandas df
    Computes distance in km
    """
    lat_1_rad, lon_1_rad = np.radians(df_in['y']), np.radians(df_in['x'])
    lat_2_rad, lon_2_rad = np.radians(df_out['y']), np.radians(df_out['x'])

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
    haversine_rad = 2 * np.arcsin(np.sqrt(a))
    haversine_km = haversine_rad * EARTH_RADIUS

    return haversine_km

import numpy as np
import pandas as pd

EARTH_RADIUS = 6371

def calc_dist(x_in, y_in, x_out, y_out ) -> float:
    """
    Calculate the haversine and Manhattan distances between two
    points on the earth (specified in decimal degrees)
    Vectorized version for pandas df
    Computes distance in km
    """
    lat_1_rad, lon_1_rad = np.radians(y_in), np.radians(x_in)
    lat_2_rad, lon_2_rad = np.radians(y_out), np.radians(x_out)

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
    haversine_rad = 2 * np.arcsin(np.sqrt(a))
    haversine_km = haversine_rad * EARTH_RADIUS

    return haversine_km

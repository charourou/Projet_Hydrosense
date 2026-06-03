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

def knn_idw(valeurs, distances) -> float:
    """
    Calcule la moyenne pondérée (IDW) pour une observation (1D).
    """
    v = np.asarray(valeurs, dtype=float)
    d = np.asarray(distances, dtype=float)

    masque = ~np.isnan(v)
    if not masque.any():
        return np.nan

    v_valides, d_valides  = v[masque], d[masque]

    # Calcul des poids (1 / d^2) avec sécurité pour une distance de 0
    poids = np.where(d_valides == 0, 1e10, 1.0 / (d_valides ** 2))
    return float(np.sum(v_valides * poids) / np.sum(poids))

def appliquer_idw_df(df: pd.DataFrame, colonnes_cibles: list, distances: list) -> pd.Series:
    """
    Applique la fonction knn_idw ligne par ligne sur un DataFrame.
    """
    # On extrait uniquement les colonnes concernées.
    series_idw = df[colonnes_cibles].apply(
        lambda ligne: knn_idw(ligne.values, distances),
        axis=1
    )

    return series_idw



if __name__ == '__main__':

    print(knn_idw([1 , 2 ,3 ],[1,5,10])
      )

    print(knn_idw([np.nan , 2 ,3 ],[1,5,10])
      )

    print(knn_idw([np.nan , np.nan ,np.nan ],[1,5,10])
      )

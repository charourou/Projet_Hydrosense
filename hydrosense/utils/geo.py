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


def trouver_voisins_hydrogeologiques(df_catalogue: pd.DataFrame, bss_target: str, n_voisins: int = 10) -> pd.DataFrame:
    """
    Trouve les piézomètres voisins les plus pertinents pour une cible donnée.
    Priorité : 1. Même Masse d'eau -> 2. Même Géologie (BD Lisa) -> 3. Distance géographique.
    """

    target_row = df_catalogue[df_catalogue['bss_id'] == bss_target]

    set_eau_target = extraire_set(target_row['codes_masse_eau_edl'].values[0])
    set_geo_target = extraire_set(target_row['codes_bdlisa'].values[0])
    lat_target, lon_target = target_row['y'].values[0], target_row['x'].values[0]

    df_candidats = df_catalogue[df_catalogue['bss_id'] != bss_target].copy()

    # Scoring hydrogéologique
    df_candidats['same_eau'] = df_candidats['codes_masse_eau_edl'].apply(lambda x: a_une_couche_commune(x, set_eau_target))
    df_candidats['same_geo'] = df_candidats['codes_bdlisa'].apply(lambda x: a_une_couche_commune(x, set_geo_target))

    # On ne garde que ceux qui ont au moins une correspondance
    df_pertinents = df_candidats[df_candidats['same_eau'] | df_candidats['same_geo']].copy()

    # Calcul de la distance
    df_pertinents['distance_km'] = df_pertinents.apply(
        lambda row: calc_dist(lat_target, lon_target, row['y'], row['x']), axis=1
    )

    # Tri multi-critères et sélection
    df_tries = df_pertinents.sort_values(
        by=['same_eau', 'same_geo', 'distance_km'],
        ascending=[False, False, True]
    )

    return df_tries.head(n_voisins)


def extraire_set(valeur_brute) -> set:
    """Transforme une chaîne de codes séparés par des virgules en un set propre."""
    if pd.isna(valeur_brute) or not valeur_brute:
        return set()
    return set([c.strip() for c in str(valeur_brute).split(',') if c.strip()])

def a_une_couche_commune(valeur_candidat, set_cible: set) -> bool:
    """Vérifie si les codes du candidat ont au moins un élément en commun avec la cible."""
    set_candidat = extraire_set(valeur_candidat)
    return len(set_cible.intersection(set_candidat)) > 0



if __name__ == '__main__':

    print(knn_idw([1 , 2 ,3 ],[1,5,10])
      )

    print(knn_idw([np.nan , 2 ,3 ],[1,5,10])
      )

    print(knn_idw([np.nan , np.nan ,np.nan ],[1,5,10])
      )

from hydrosense.database.meteo import CatalogueMeteo
from hydrosense.utils.geo import calc_dist, appliquer_idw_df
from hydrosense.database.bigquery import info_piezo
from hydrosense.utils.evap import estim_PU

import pandas as pd
import numpy as np


VOISINS_DEPARTEMENTS = {
        '16': ['24', '79', '17', '33', '87', '86'] ,
        '17': ['85', '79', '16', '33'] ,
        '24': ['33', '16', '87' , '47'] ,
        '33': ['17', '16', '24', '47', '40'],
        '35': ['44', '53'],
        '40': ['33', '47', '32', '64'],
        '44': ['56', '35', '49', '85'],
        '47': ['33', '24', '40'],
        '79': ['85', '49', '86', '16', '17'],
        '85': ['44', '49', '79', '17'],
        '82': ['47', '32', '46'],
        '86': ['79', '16', '87'],
        '87': ['16', '24', '86']
    }


class SynthPrecipitation():
    """
    Classe qui permet la création d'un nouveau pd.Dataframe
    avec de la donnée météorologique concomitant à un dataframe de pizeo.
    """

    def __init__(self, df_piezo: pd.DataFrame, bss_piezo: str):
        """
        Reconnait un dataframe de piezometrie nettoyé qui a un pas de temps régulier et quotidien
        example : le bourdet BSS001QHYH
        """
        self.df_piezo = df_piezo.copy()
        self.bss_id = bss_piezo
        self.info = info_piezo(bss_piezo)

        try:
            self.df_piezo['date_mesure'] = pd.to_datetime(self.df_piezo['date_mesure'])
        except:
            print("Aucune colonne de date trouvée dans df_piezo.")


    def search(self, code_dep:str, n_neighbors = 3, dep_voisins = False):
        """
        Utilise le module database.meteo pour charger un dictionnaire de dataframe avec
        des données météo.
        Si n_neighbors = 1, choisi le plus proche géographiquement. see calc_dist()

        Retourne un dataframe meteo interpolé sur index de dates du dataframe piezo
        """

        deps_a_chercher = [code_dep]
        if dep_voisins and code_dep in VOISINS_DEPARTEMENTS:
            deps_a_chercher.extend(VOISINS_DEPARTEMENTS[code_dep])
            print(f"🌍 Recherche ETENDUE (avec voisins) pour : {', '.join(deps_a_chercher)}...")
        else:
            print(f"🌍 Recherche STRICTE limitée au département : {code_dep}...")

        print(f"🌍 Recherche des stations météo pour le département {code_dep}...")

        meteo = CatalogueMeteo()

        # Grand pool de stations
        pool_stations_global = {}
        for dep in deps_a_chercher:

            meteo = CatalogueMeteo() # instance neuve
            dict_meteo_dept = meteo.extraire_departement(dep)
            if dict_meteo_dept:
                pool_stations_global.update(dict_meteo_dept)

        if not pool_stations_global:
            print("❌ Aucune donnée météo récupérée dans la zone.")
            return {}

        # Récupération des coordonnées du piézomètre grace à la fonction info_piezo
        x_piezo, y_piezo = self.info.loc[0,'x'], self.info.loc[0,'y']


        # Calcul des distances pour chaque station météo
        distances_stations = []
        for id_station, df_m in dict_meteo_dept.items():
            if df_m.empty:
                continue

            x_station, y_station = df_m['LON'].iloc[0], df_m['LAT'].iloc[0]
            distance_km = calc_dist(x_piezo, y_piezo, x_station, y_station)

            distances_stations.append({
                'id_station': id_station,
                'distance': distance_km,
                'dataframe': df_m
            })

        distances_stations.sort(key=lambda x: x['distance'])
        top_stations = distances_stations[:n_neighbors]

        resultat = {}
        for item in top_stations:
            print(f"🎯 Station météo retenue : {item['id_station']} (à {item['distance']:.2f} km)")

        # On stocke le dataframe ET la distance pour le calcul des poids
            resultat[item['id_station']] = {
                'dataframe': item['dataframe'],
                'distance': item['distance']
            }

        return resultat


    def merge(self, dict_meteo):
        """
        concatenatation de la meteo et du piezo
        """

        df_final = self.df_piezo.copy()

        # Listes pour stocker l'ordre exact des stations et de leurs distances
        distances_ordonnees = []
        ids_stations = []

        for id_station, infos in dict_meteo.items():
            df_m_clean = infos['dataframe'].copy()
            distances_ordonnees.append(infos['distance'])
            ids_stations.append(id_station)

            # On force la date météo en datetime64 pour la jointure
            df_m_clean['date_RR'] = pd.to_datetime(df_m_clean['date_RR'])

            # On ne garde que la date et les variables utiles pour ne pas polluer le tableau final
            colonnes_utiles = ['date_RR', 'RR', 'TM', 'FFM']
            colonnes_existantes = [c for c in colonnes_utiles if c in df_m_clean.columns]
            df_m_clean = df_m_clean[colonnes_existantes]

            # Renommage dynamique pour identifier la station (ex: RR devient RR_17132001)
            renommage = {c: f"{c}_{id_station}" for c in ['RR', 'TM', 'FFM'] if c in df_m_clean.columns}
            df_m_clean.rename(columns=renommage, inplace=True)

            # Jointure à GAUCHE (left merge) : S'il manque de la météo ce jour-là, ça fera un NaN.
            df_final = df_final.merge(
                df_m_clean,
                left_on='date_mesure',
                right_on='date_RR',
                how='left'
            )

            df_final.drop(columns=['date_RR'], inplace=True, errors='ignore')

        # ------
        print("\n⚙️ Calcul des variables synthétiques (Pondération IDW)...")
        variables_climatiques = ['RR', 'TM', 'FFM']

        for var in variables_climatiques:
            # On liste les colonnes fusionnées (ex: ['RR_17132001', 'RR_17154002', ...])
            cols_var = [f"{var}_{id_st}" for id_st in ids_stations if f"{var}_{id_st}" in df_final.columns]

            if cols_var:
                # Application de notre fonction vectorisée et On arrondit à 2 décimales
                df_final[f'{var}_synth'] = appliquer_idw_df(df_final, cols_var, distances_ordonnees)
                df_final[f'{var}_synth'] = df_final[f'{var}_synth'].round(2)

        return df_final

    def calc_PU(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """
        Prend le DataFrame issu de la méthode merge(), calcule la Pluie Utile
        et ajoute la colonne 'PU_synth'.
        """
        print("\n💧 Calcul de la Pluie Utile (Penman-Monteith)...")
        df = df_merged.copy()

        colonnes_requises = ['RR_synth', 'TM_synth', 'FFM_synth']
        for col in colonnes_requises:
            if col not in df.columns:
                print(f"❌ Erreur : Colonne {col} manquante. Veuillez exécuter .merge() d'abord.")
                return df

        try:
            lat = float(self.info['y'].iloc[0])
            altitude = float(self.info['altitude_station'].iloc[0])
        except Exception as e:
            print(f"⚠️ Impossible d'extraire la latitude ou l'altitude. Valeurs par défaut appliquées. ({e})")
            lat, altitude = 46.0, 100.0  # Valeurs moyennes en France par sécurité


        df['date_mesure'] = pd.to_datetime(df['date_mesure'])
        jours_annee = df['date_mesure'].dt.dayofyear

        # S'il manque de la pluie un jour, c'est 0. Pour la température/vent, on interpole.
        P_mm = df['RR_synth'].fillna(0)
        T_moy_C = df['TM_synth'].interpolate(method='linear').fillna(12.0)
        vent_m_s = df['FFM_synth'].interpolate(method='linear').fillna(2.0)

        df['PU_synth'] = estim_PU(
            P_mm=P_mm,
            T_moy_C=T_moy_C,
            vent_m_s=vent_m_s,
            lat_deg=lat,
            altitude_m=altitude,
            jour_annee=jours_annee
        )

        return df


if __name__ == '__main__':

    test_file = '/home/charourou/projects/Projet_Hydrosense/data/piezo_bourdet_clean.csv'
    df_piezo = pd.read_csv(test_file, sep=';')

    synthetiseur = SynthPrecipitation(df_piezo, 'BSS001QHYH')
    departement = '79'
    resultat = synthetiseur.search(departement)
    print(resultat)

    df_final = synthetiseur.merge(resultat)
    print(df_final)

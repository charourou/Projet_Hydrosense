from hydrosense.database.meteo import CatalogueMeteo
from hydrosense.utils.geo import calc_dist
import pandas as pd
import numpy as np


class SynthPrecipitation():
    """
    Class qui permet la création d'un pd. Dataframe avec de la donnée météorologique concomitant à un dataframe de pizeo.

    """

    def __init__(self, df_piezo):
        """
        Reconnait un dataframe de piezometrie nettoyé qui a un pas de temps régulier et quotidien

        """
        self.df_piezo = df_piezo





    def search(self, code_dep, n_neighbors = 1):
        """
        Utilise le module database.meteo pour charger un dictionnaire de dataframe avec
        des données météo.
        Si n_neighbors = 1, choisi le plus proche géographiquement. see calc_dist()

        Retourne un dataframe meteo interpolé sur index de dates du dataframe piezo
        """

        id_station = '44100011' # a éditer en fonction de la station météo


        return dict(id_station = pd.DataFrame())



    def merge(self, dict_meteo):
        """
        concatenatation of the meteo information contained in the dict_meteo with self.pizeo


        """

        return pd.DataFrame()

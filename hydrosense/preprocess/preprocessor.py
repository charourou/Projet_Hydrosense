import numpy as np
import pandas as pd
from hydrosense import params
#from colorama import Fore, Styl

def preprocess_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rééchantillonne les données journalières en moyennes mensuelles
    - Construit les features de lag et moyennes mobiles pour XGBoost
    - Supprime les lignes avec NaN (dues aux shifts)

    Features produites :
        Semaine        → saisonnalité semaine
        lag_1/2/3/4   → niveaux des 4 semaines précédents
        lag_52      → niveau même semaine l'année précédente
        moyenne_3w  → tendance récente (3 mois)
        moyenne_6w  → tendance moyen terme (6 mois)
    """
    #print(Fore.MAGENTA +
    print("\n⭐️ Use case: preprocess" ) #+ Style.RESET_ALL)


    if 'date_mesure' in df.columns:
        df.set_index('date_mesure', inplace=True)

    # S'assure que l'index est bien de type DatetimeIndex
    df.index = pd.to_datetime(df.index)


    df_resample = df.resample('W').agg({
        params.TARGET_COL: 'mean',   # niveau nappe → moyenne
        'RR_synth': 'sum'            # précipitations → cumul hebdomadaire
    })

    df_w = pd.DataFrame(df_resample)

    df_w['semaine'] = df_w.index.isocalendar().week

    df_w['lag_1'] = df_w['niveau_nappe_eau'].shift(1)
    df_w['lag_2'] = df_w['niveau_nappe_eau'].shift(2)
    df_w['lag_3'] = df_w['niveau_nappe_eau'].shift(3)
    df_w['lag_4'] = df_w['niveau_nappe_eau'].shift(4)
    df_w['lag_52'] = df_w['niveau_nappe_eau'].shift(52)
    df_w['RR_synth'] = df_w['RR_synth']
    #df_w['RR_lag_2'] = df_w['RR_synth'].shift(2)   # pluie il y a 2 semaines
    #df_w['RR_moy_4w'] = df_w['RR_synth'].rolling(window=4).sum()  # cumul 4 semaines

    # Moyenne du niveau de la nappe sur les 3 derniers mois (tendance récente)
    df_w['moyenne_3w'] = df_w['niveau_nappe_eau'].shift(1).rolling(window=3).mean()

    # Moyenne du niveau sur les 6 derniers mois (tendance moyen terme)
    df_w['moyenne_6w'] = df_w['niveau_nappe_eau'].shift(1).rolling(window=6).mean()

    # IMPORTANT : Applique le .dropna() APRÈS avoir créé ces nouvelles variables
    df_w = df_w.dropna()

    X = df_w[['semaine', 'lag_1', 'lag_2', 'lag_3','lag_4', 'lag_52', 'moyenne_3w', 'moyenne_6w','RR_synth']]

    y_target = df_w['niveau_nappe_eau']

    print(f"✅ preprocess() done — {len(df_w)} semaines | {df_w.shape[1]} colonnes\n")
    return df_w

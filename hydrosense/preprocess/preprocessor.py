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


def preprocess_week_w_PU_synth(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rééchantillonne les données journalières en données hebdomadaires ('W')
    - Supprime la variable 'semaine' brute au profit de semaine_sin / cos
    - Génère les lags de la target (1, 2, 3, 4, 52) et ses moyennes mobiles (3w, 6w)
    - Génère le lag 1 pour PC1, PC2, PC3 puis supprime les colonnes d'origine
    - Génère les lags 1, 2, 3, 4 pour PU_synth puis supprime la colonne d'origine
    - Supprime les lignes contenant des NaN induits par les décalages (shifts)
    """
    print("\n⭐️ Use case: preprocess_week_w_PU_synth")

    # Copie locale pour éviter les warnings "SettingWithCopyWarning"
    df_local = df.copy()

    if 'date_mesure' in df_local.columns:
        df_local.set_index('date_mesure', inplace=True)

    # S'assure que l'index est bien de type DatetimeIndex
    df_local.index = pd.to_datetime(df_local.index)

    # 1. RÉÉCHANTILLONNAGE HEBDOMADAIRE ('W')
    agg_dict = {
        'niveau_nappe_eau': 'mean'
    }

    # Intégration de PU_synth, PC1, PC2, PC3 à l'agrégation hebdomadaire (si présents)
    cols_to_lag = ['PU_synth', 'PC1', 'PC2', 'PC3']
    for col_name in cols_to_lag:
        if col_name in df_local.columns:
            agg_dict[col_name] = 'mean'

    df_w = df_local.resample('W').agg(agg_dict)

    # 2. ENCODAGE CYCLIQUE DE LA SAISONNALITÉ (Sinus / Cosinus)
    semaine_brute = df_w.index.isocalendar().week.astype(int)
    df_w['semaine_sin'] = np.sin(2 * np.pi * semaine_brute / 52)
    df_w['semaine_cos'] = np.cos(2 * np.pi * semaine_brute / 52)

    # 3. FEATURES DE LAGS & ROLLING (Sur la Target 'niveau_nappe_eau')
    df_w['lag_1'] = df_w['niveau_nappe_eau'].shift(1)
    df_w['lag_2'] = df_w['niveau_nappe_eau'].shift(2)
    df_w['lag_3'] = df_w['niveau_nappe_eau'].shift(3)
    df_w['lag_4'] = df_w['niveau_nappe_eau'].shift(4)
    df_w['lag_52'] = df_w['niveau_nappe_eau'].shift(52)


    # 4. FEATURES SUR LES COMPOSANTES PRINCIPALES (Lag 1 uniquement)
    for pc in ['PC1', 'PC2', 'PC3']:
        if pc in df_w.columns:
            df_w[f'{pc}_lag_1'] = df_w[pc].shift(1)

    # 5. FEATURES SUR PU_SYNTH (Lags 1, 2, 3, 4)
    if 'PU_synth' in df_w.columns:
        for lag in [1, 2, 3, 4]:
            df_w[f'PU_synth_lag_{lag}'] = df_w['PU_synth'].shift(lag)

    # 6. 🗑️ NETTOYAGE DES COLONNES INJECTÉES (Elles doivent disparaître !)
    # On ne garde que les versions décalées (lags) pour interdire toute fuite de données
    cols_to_drop = [col for col in cols_to_lag if col in df_w.columns]
    df_w = df_w.drop(columns=cols_to_drop)

    # 7. NETTOYAGE DES NaN
    df_w = df_w.dropna()

    print(f"✅ preprocess() done — {len(df_w)} semaines | {df_w.shape[1]} colonnes\n")

    return df_w

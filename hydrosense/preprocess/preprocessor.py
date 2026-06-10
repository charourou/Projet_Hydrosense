import numpy as np
import pandas as pd
from hydrosense import params

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    """
    #print(Fore.MAGENTA +
    print("\n⭐️ Use case: preprocess" ) #+ Style.RESET_ALL)


    if 'date_mesure' in df.columns:
        df.set_index('date_mesure', inplace=True)

    # S'assure que l'index est bien de type DatetimeIndex
    df.index = pd.to_datetime(df.index)


    df_resample = df.resample('W').agg({
        params.TARGET_COL: 'mean',    # niveau nappe → moyenne
        'RR_synth': 'sum',            # précipitations → cumul hebdomadaire
        'PU_synth': 'sum',
        'TM_synth': 'mean',
        'PC1': 'mean',
        'PC2': 'mean',
        'PC3': 'mean',
        'FFM_synth': 'mean'
    })

    df_w = pd.DataFrame(df_resample)

    df_w['semaine'] = df_w.index.isocalendar().week
    df_w['semaine_sin'] = np.sin(2 * np.pi * df_w['semaine'] / 52)
    df_w['semaine_cos'] = np.cos(2 * np.pi * df_w['semaine'] / 52)
    df_w = df_w.drop(['semaine'], axis=1)
    print(df_w.head())

    if False:
        df_w['lag_1'] = df_w['niveau_nappe_eau'].shift(1)
        df_w['lag_2'] = df_w['niveau_nappe_eau'].shift(2)
        df_w['lag_3'] = df_w['niveau_nappe_eau'].shift(3)
        df_w['lag_4'] = df_w['niveau_nappe_eau'].shift(4)
        df_w['lag_52'] = df_w['niveau_nappe_eau'].shift(52)
        df_w['RR_synth'] = df_w['RR_synth']
    #df_w['RR_lag_2'] = df_w['RR_synth'].shift(2)   # pluie il y a 2 semaines
    #df_w['RR_moy_4w'] = df_w['RR_synth'].rolling(window=4).sum()  # cumul 4 semaines

    if False:
        df_w['moyenne_3w'] = df_w['niveau_nappe_eau'].shift(1).rolling(window=3).mean()
        df_w['moyenne_6w'] = df_w['niveau_nappe_eau'].shift(1).rolling(window=6).mean()

    # IMPORTANT : Applique le .dropna() APRÈS avoir créé ces nouvelles variables
    df_w = df_w.dropna()

    # X = df_w[['semaine', 'lag_1', 'lag_2', 'lag_3','lag_4', 'lag_52', 'moyenne_3w', 'moyenne_6w','RR_synth']]
    # y_target = df_w['niveau_nappe_eau']

    print(f"✅ preprocess() done — {len(df_w)} semaines | {df_w.shape[1]} colonnes\n")
    return df_w

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def scale_feats(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple:
    """
    Adimensionne les variables explicatives pour le modèle.
    Entraîne le StandardScaler sur le Train et l'applique sur le Train et le Test.

    Returns
    -------
    X_train_scaled : pd.DataFrame standardisé
    X_test_scaled  : pd.DataFrame standardisé
    scaler         : L'objet StandardScaler entraîné (utile pour inverser ou pour les prédictions futures)
    """
    print("Adimensionnement des features (Standardisation)...")

    feat_to_scale = ['niveau_nappe_eau', 'TM_synth', 'PC1','PC2', 'PC3', 'PU_synth']
    feat_to_minmax = ['RR_synth','FFM_synth']

    # cols_std = [c for c in feat_to_scale if c in X_train_df.columns]
    cols_std = []
    for col in X_train_df.columns:
        is_lagged_feature = any(col.startswith(f"{f}_lag_") for f in feat_to_scale)
        is_base_feature = col in feat_to_scale
        if is_base_feature or is_lagged_feature:
            cols_std.append(col)

    cols_mm = [c for c in feat_to_minmax if c in X_train_df.columns]

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    minmaxer = MinMaxScaler()
    minmaxer.set_output(transform="pandas")


    # 3. Fit & Transform sur le Train, Transform sur le Test
    X_train_out = X_train_df.copy()
    X_test_out = X_test_df.copy()

    X_train_out[cols_std] = scaler.fit_transform(X_train_df[cols_std])
    X_test_out[cols_std] = scaler.transform(X_test_df[cols_std])

    if cols_mm:
        X_train_out[cols_mm] = minmaxer.fit_transform(X_train_df[cols_mm])
        X_test_out[cols_mm] = minmaxer.transform(X_test_df[cols_mm])


    print(f"✅ Adimensionnement terminé : {len(cols_std)} en Standard, {len(cols_mm)} en MinMax.")
    return X_train_out, X_test_out, scaler, minmaxer


import pandas as pd


def prepare_lags(df_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les lags (1, 2, 3, 4 et 52) pour les colonnes spécifiques :
    'PU', 'PC' et 'TARGET_COL'.
    """

    df_lagged = df_scaled.copy()

    # Configuration des lags par feature
    lag_config = {
        params.TARGET_COL: [1, 2, 3, 4, 52],
        "PC1": [1],
        "PC2": [1],
        "PC3": [1],
        "PU_synth": [1, 2, 3, 4],
    }

    new_columns = {}

    for col, lags in lag_config.items():
        if col in df_lagged.columns:
            for lag in lags:
                new_columns[f"{col}_lag_{lag}"] = df_lagged[col].shift(lag)
        else:
            print(f"Attention : La colonne '{col}' est absente du DataFrame.")


    if new_columns:
        df_new_features = pd.DataFrame(new_columns, index=df_lagged.index)
        df_lagged = pd.concat([df_lagged, df_new_features], axis=1)

        df_lagged.drop(lag_config.keys(), axis=1, inplace=True)

    df_lagged.dropna(inplace=True)
    return df_lagged


if __name__ == "__main__":

    # Pipeline exemple
    df = load_plean(DATA_CODE_PIEZO)
    df_w = preprocess_week(df)

    # Lagging
    X_lagged = prepare_lags(df_w)


    # On split ensuite.
    print(X_lagged.columns)
    X_train, X_test= split_lagged_data(X_lagged)
    # On a perdu le y )
    _, _, y_train, y_test = split_data(df_w)

    # Scaling
    X_train_scaled, X_test_scaled, scaler, _ = scale_feats(X_train_df = X_train, X_test_df= X_test)

import numpy as np
import pandas as pd
from hydrosense import params

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from colorama import Fore, Style


from hydrosense.database.bigquery import load_plean
from typing import Tuple

from hydrosense.interface.main import split_lagged_data, split_data


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

    # IMPORTANT : Applique le .dropna() APRÈS avoir créé ces nouvelles variables
    df_w = df_w.dropna()

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

    try:
        X_train_out[cols_std] = scaler.fit_transform(X_train_df[cols_std])
    except:
        print('error')
        print(cols_std)
        print(X_train_df)
        raise

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


def make_preproc_week(df: pd.DataFrame):
    """
    Pipeline complet : Prétraitement -> Lags -> Split -> Scaling
    Retourne X_train, X_test, y_train, y_test scalés.
    """

    si = SimpleImputer(strategy='constant', fill_value = 0)
    si.set_output(transform="pandas")

    cols_pc = list(filter(lambda x: x.startswith('PC'), df.columns))
    df[cols_pc] = si.fit_transform(df[cols_pc])


    # Nettoyage & Feature Engineering (Lags)
    df_w = preprocess_week(df)
    df_full = prepare_lags(df_w)

    # 2. Split temporel (on utilise la date pour découper proprement)
    # il va falloir utiliser ce pipeline pour la production ou faire un

    X_train, X_test= split_lagged_data(df_full)
    _, _, y_train, y_test = split_data(df_w)

    assert X_train.shape[0] > 0

    X_train_scaled, X_test_scaled, scaler, _ = scale_feats( X_train_df=X_train,
                                                       X_test_df=X_test
                                                        )
    y_train = y_train.loc[X_train_scaled.index]

    # Vérification critique
    assert X_train_scaled.shape[0] == y_train.shape[0], "Mismatch entre X_train et y_train !"

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler





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
    # il faut clipper les données y_train
    y_train = y_train.loc[X_train.index]


    # Scaling
    X_train_scaled, X_test_scaled, scaler, _ = scale_feats(X_train_df = X_train, X_test_df= X_test)

    assert X_train_scaled.shape[0] == y_train.shape[0]

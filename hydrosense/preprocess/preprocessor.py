import numpy as np
import pandas as pd
from hydrosense import params

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from colorama import Fore, Style

from hydrosense.database.bigquery import load_plean
from typing import Tuple


def preprocess_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rééchantillonne les données journalières en moyennes mensuelles
    - Construit les features de lag et moyennes mobiles pour XGBoost
    - Supprime les lignes avec NaN (dues aux shifts)

    Features produites :
        Semaine sin et cos       → saisonnalité semaine

    """

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

    print(f"✅ Moyenne hebdomadaire — {len(df_w)} semaines | {df_w.shape[1]} colonnes\n")
    return df_w

def split_lagged_data(df_lagged: pd.DataFrame,
                      feature_cols: list,  target_col: str,
                      train_end: str,test_start: str,test_end: str):
    """Splits lagged data into train and test sets for features."""

    cols_to_retain = feature_cols + [target_col]

    selected_features = []
    for col in df_lagged.columns:
        is_base_feature = col in feature_cols
        is_lagged_feature = any(
            col.startswith(f"{f}_lag_") for f in cols_to_retain
        )
        if is_base_feature or is_lagged_feature:
            selected_features.append(col)

    X = df_lagged[selected_features]

    X_train_df = X.loc[:train_end]
    X_test_df = X.loc[test_start:test_end]

    return X_train_df, X_test_df

def split_data(df_ml: pd.DataFrame,
               feature_cols: list, target_col: str,
               train_end: str, test_start: str, test_end: str):
    """
    Découpe en X_train, X_test, y_train, y_test.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: split_data" + Style.RESET_ALL)

    cols = [c for c in df_ml.columns if c in feature_cols]

    X = df_ml[cols]
    y = df_ml[target_col]

    X_train_df = X.loc[:train_end]
    X_test_df  = X.loc[test_start:test_end]
    y_train_df = y.loc[:train_end]
    y_test_df  = y.loc[test_start:test_end]

    print(f"✅ split_data() done — Train : {len(X_train_df)} | Test : {len(X_test_df)}\n")
    return X_train_df, X_test_df, y_train_df, y_test_df



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


def make_preproc_week(df: pd.DataFrame,
                      feature_cols: list, target_col: str,
                      train_end: str, test_start: str, test_end: str):
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

    X_train, X_test = split_lagged_data(
        df_full, feature_cols, target_col, train_end, test_start, test_end
    )
    _, _, y_train, y_test = split_data(
        df_w, feature_cols, target_col, train_end, test_start, test_end
    )

    assert X_train.shape[0] > 0

    X_train_scaled, X_test_scaled, scaler, _ = scale_feats( X_train_df=X_train,
                                                       X_test_df=X_test
                                                        )
    y_train = y_train.loc[X_train_scaled.index]

    # Vérification critique
    assert X_train_scaled.shape[0] == y_train.shape[0], "Mismatch entre X_train et y_train !"

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



if __name__ == "__main__":

    # Définition des constantes pour le test
    DATA_CODE_PIEZO = "BSS001QHYH"
    FEATURE_COLS = ["semaine_sin", "semaine_cos", "PU_synth", "PC1", "PC2", "PC3"]
    TRAIN_END = "2025-02-28"
    TEST_START = "2025-03-01"
    TEST_END = "2025-05-31"

    # Pipeline exemple
    df = load_plean(DATA_CODE_PIEZO)
    df_w = preprocess_week(df)

    # Lagging
    X_lagged = prepare_lags(df_w)

    # On split ensuite.
    print(X_lagged.columns)
    X_train, X_test = split_lagged_data(
        X_lagged, FEATURE_COLS, params.TARGET_COL, TRAIN_END, TEST_START, TEST_END
    )
    # On a perdu le y )
    _, _, y_train, y_test = split_data(
        df_w, FEATURE_COLS, params.TARGET_COL, TRAIN_END, TEST_START, TEST_END
    )
    # il faut clipper les données y_train
    y_train = y_train.loc[X_train.index]

    # Scaling
    X_train_scaled, X_test_scaled, scaler, _ = scale_feats(X_train_df = X_train, X_test_df= X_test)

    assert X_train_scaled.shape[0] == y_train.shape[0]

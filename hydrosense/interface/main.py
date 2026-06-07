import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model
from hydrosense.ml_logic.folding import get_folds

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo

# ══════════════════════════════════════════════════════════════════════════════
# PARAMS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = Path("data/piezo_bourdet_clean.csv")
DATA_CODE_PIEZO = "BSS001QHYH"
TARGET_COL   = "niveau_nappe_eau"
DATE_COL     = "date_mesure"
FEATURE_COLS = ["mois", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]

# Split : 3 derniers mois en test (Mars → Mai 2026)
TRAIN_END  = "2026-02-28"
TEST_START = "2026-03-01"
TEST_END   = "2026-05-31"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Charge le CSV brut et retourne un DataFrame avec un DatetimeIndex.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: load_data" + Style.RESET_ALL)

    df = pd.read_csv(path, parse_dates=[DATE_COL], sep = ';')

    if DATE_COL in df.columns:
        df.set_index(DATE_COL, inplace=True)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    print(f"✅ load_data() done — {len(df)} rows | {df.index.min().date()} → {df.index.max().date()}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Rééchantillonne les données journalières en moyennes mensuelles
    - Construit les features de lag et moyennes mobiles pour XGBoost
    - Supprime les lignes avec NaN (dues aux shifts)

    Features produites :
        mois        → saisonnalité mensuelle
        lag_1/2/3   → niveaux des 3 mois précédents
        lag_12      → niveau même mois l'année précédente
        moyenne_3m  → tendance récente (3 mois)
        moyenne_6m  → tendance moyen terme (6 mois)
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: preprocess" + Style.RESET_ALL)

    # Rééchantillonnage mensuel — 'ME' = Month End (pandas >= 2.2)
    # y_mensuel = df[TARGET_COL].resample("ME").mean()
    y_mensuel = df.resample("ME", on="date_mesure")[TARGET_COL].mean()

    # Feature engineering
    df_ml = pd.DataFrame(y_mensuel)
    df_ml["mois"]       = df_ml.index.month
    df_ml["lag_1"]      = df_ml[TARGET_COL].shift(1)
    df_ml["lag_2"]      = df_ml[TARGET_COL].shift(2)
    df_ml["lag_3"]      = df_ml[TARGET_COL].shift(3)
    df_ml["lag_12"]     = df_ml[TARGET_COL].shift(12)
    df_ml["moyenne_3m"] = df_ml[TARGET_COL].rolling(window=3).mean()
    df_ml["moyenne_6m"] = df_ml[TARGET_COL].rolling(window=6).mean()
    df_ml = df_ml.dropna()

    print(f"✅ preprocess() done — {len(df_ml)} mois | {df_ml.shape[1]} colonnes\n")
    return df_ml


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df_ml: pd.DataFrame):
    """
    Découpe en X_train, X_test, y_train, y_test.
    Test = Mars → Mai 2026 (3 derniers mois).
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: split_data" + Style.RESET_ALL)

    X = df_ml[FEATURE_COLS]
    y = df_ml[TARGET_COL]

    X_train_df = X.loc[:TRAIN_END]
    X_test_df  = X.loc[TEST_START:TEST_END]
    y_train_df = y.loc[:TRAIN_END]
    y_test_df  = y.loc[TEST_START:TEST_END]

    print(f"✅ split_data() done — Train : {len(X_train_df)} mois | Test : {len(X_test_df)} mois\n")
    return X_train_df, X_test_df, y_train_df, y_test_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train_df: pd.DataFrame, y_train_df: pd.Series, optimize: bool = True):
    """
    Optimise et entraîne le modèle XGBoost.

    Parameters
    ----------
    X_train, y_train : données d'entraînement
    optimize         : True → GridSearchCV | False → hyperparamètres par défaut

    Returns
    -------
    (model, history)
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)

    if optimize:
        model, best_params = optimize_model(
            X_train_df.values, # Les valeurs NumPy pour l'entraînement du modèle
            y_train_df.values, # Les valeurs NumPy pour l'entraînement du modèle
            X_train_df_for_folds=X_train_df, # Le DataFrame pour la génération des folds
            # date_column_name=DATE_COL,       # Plus nécessaire car optimize_model ne le passe plus à get_folds
            n_splits_cv=3
        )
        print(Fore.BLUE + f"\nBest params: {best_params}" + Style.RESET_ALL)
    else:
        model = initialize_model()

    model, history = train_model(model, X_train_df.values, y_train_df.values)

    print("✅ train() done \n")
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test) -> dict:
    """
    Évalue le modèle sur les 3 mois de test (jamais vus à l'entraînement).

    Returns
    -------
    metrics : dict {"mae": float, "rmse": float, "r2": float}
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    metrics = evaluate_model(model, X_test, y_test)

    print("✅ evaluate() done \n")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICT — prévision 3 mois futurs (pour Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def pred(model, df_ml: pd.DataFrame) -> pd.Series:
    """
    Génère les prévisions sur les 3 prochains mois.
    Utilisé par Streamlit (app/main.py) pour afficher les prévisions futures.

    Returns
    -------
    pd.Series avec 3 valeurs prédites et leurs dates (index DatetimeIndex)
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: pred" + Style.RESET_ALL)

    X_future       = df_ml[FEATURE_COLS].tail(3).values
    y_pred         = predict_model(model, X_future)
    last_date      = df_ml.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=4, freq="ME")[1:]
    forecast       = pd.Series(y_pred, index=forecast_index, name=TARGET_COL)

    print(f"\n✅ pred() done:\n{forecast.to_string()}\n")
    return forecast


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — données chargées une seule fois, modèle partagé
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Données — une seule fois
    #df    = load_data() # du CSV
    # load from big query
    df = clean_piezo(load_piezo_bq(DATA_CODE_PIEZO))

    df_ml = preprocess(df)
    X_train_df, X_test_df, y_train_df, y_test_df = split_data(df_ml)

    # --- Zone de test pour visualiser les splits de cross-validation ---
    print(Fore.CYAN + "\n--- Visualisation des splits de cross-validation annuels ---" + Style.RESET_ALL)
    try:
        # get_folds attend un DataFrame avec la colonne de dates.
        test_splits = get_folds(
            # Passe directement le DatetimeIndex de X_train_df
            dates_series=X_train_df.index,
            n_splits=3, # Nombre de splits à visualiser
            min_train_years=3,
            val_years_duration=1
        )
        for i, (train_idx, val_idx) in enumerate(test_splits):
            # Utilise X_train_df pour récupérer les dates réelles pour la visualisation
            train_years = X_train_df.iloc[train_idx].index.year.unique()
            val_years = X_train_df.iloc[val_idx].index.year.unique()
            print(f"  Split {i+1}:")
            print(f"    Train years: {min(train_years)}-{max(train_years)} ({len(train_idx)} samples)")
            print(f"    Val years:   {min(val_years)}-{max(val_years)} ({len(val_idx)} samples)")
            # Assertions pour vérifier l'absence de fuite temporelle et l'ordre chronologique
            assert max(train_years) < min(val_years), f"Fuite temporelle détectée dans Split {i+1}!"
            assert max(train_idx) < min(val_idx), f"Indices non chronologiques ou chevauchement détecté dans Split {i+1}!"
    except ValueError as e:
        print(f"  Erreur lors de la visualisation des splits: {e}")
    print(Fore.CYAN + "--- Fin de la visualisation des splits ---" + Style.RESET_ALL)

    # 2. Train — optimize=True pour GridSearchCV
    model, history = train(X_train_df, y_train_df, optimize=True)

    # 3. Evaluate — même modèle, pas de rechargement
    metrics = evaluate(model, X_test_df.values, y_test_df.values)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

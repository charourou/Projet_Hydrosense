import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model

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

    df = pd.read_csv(path, parse_dates=[DATE_COL])

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
    y_mensuel = df[TARGET_COL].resample("ME").mean()

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

    X_train = X.loc[:TRAIN_END].values
    X_test  = X.loc[TEST_START:TEST_END].values
    y_train = y.loc[:TRAIN_END].values
    y_test  = y.loc[TEST_START:TEST_END].values

    print(f"✅ split_data() done — Train : {len(X_train)} mois | Test : {len(X_test)} mois\n")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(X_train, y_train, optimize: bool = True):
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
        model, best_params = optimize_model(X_train, y_train, cv=3)
        print(Fore.BLUE + f"\nBest params: {best_params}" + Style.RESET_ALL)
    else:
        model = initialize_model()

    model, history = train_model(model, X_train, y_train)

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
    X_train, X_test, y_train, y_test = split_data(df_ml)

    # 2. Train — optimize=True pour GridSearchCV
    model, history = train(X_train, y_train, optimize=True)

    # 3. Evaluate — même modèle, pas de rechargement
    metrics = evaluate(model, X_test, y_test)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

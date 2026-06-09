import time


import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo, clean_piezo2
from hydrosense.preprocess.preprocessor import preprocess_week

from hydrosense import params


# ══════════════════════════════════════════════════════════════════════════════
# PARAMS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = Path("data/piezo_bourdet_clean.csv")
#DATA_CODE_PIEZO = "BSS001QHYH"
DATA_CODE_PIEZO = "BSS001QTKG"

DATE_COL     = "date_mesure"
#FEATURE_COLS = ["mois", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]
#FEATURE_COLS = ["semaine", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]
#FEATURE_COLS = ["semaine", "lag_1", "lag_2", "lag_3","lag_4" ,"lag_52", "moyenne_3w", "moyenne_6w","RR_lag_1","RR_lag_2","RR_moy_4w"]
FEATURE_COLS = ["semaine_sin","semaine_cos", "lag_1", "lag_2", "lag_3","lag_4" ,"lag_52", "moyenne_3w", "moyenne_6w","RR_synth"]
#FEATURE_COLS = ["semaine_sin","semaine_cos", "lag_1","lag_4" ,"lag_52", "moyenne_3w", "moyenne_6w","RR_lag_1","RR_lag_2","RR_moy_4w"]

# Split : 3 derniers mois en test (Mars → Mai 2026)
TRAIN_END  = "2026-02-28"
#TEST_START = "2026-03-01" # EVALUATION_START_DATE dans params
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
    y_mensuel = df[params.TARGET_COL].resample("ME").mean()

    # Feature engineering
    df_ml = pd.DataFrame(y_mensuel)
    df_ml["mois"]       = df_ml.index.month
    df_ml["lag_1"]      = df_ml[params.TARGET_COL].shift(1)
    df_ml["lag_2"]      = df_ml[params.TARGET_COL].shift(2)
    df_ml["lag_3"]      = df_ml[params.TARGET_COL].shift(3)
    df_ml["lag_12"]     = df_ml[params.TARGET_COL].shift(12)
    df_ml["moyenne_3m"] = df_ml[params.TARGET_COL].rolling(window=3).mean()
    df_ml["moyenne_6m"] = df_ml[params.TARGET_COL].rolling(window=6).mean()
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
    print(FEATURE_COLS)
    y = df_ml[params.TARGET_COL]

    X_train = X.loc[:TRAIN_END].values
    X_test  = X.loc[params.EVALUATION_START_DATE:TEST_END].values
    y_train = y.loc[:TRAIN_END].values
    y_test  = y.loc[params.EVALUATION_START_DATE:TEST_END].values

    print(f"✅ split_data() done — Train : {len(X_train)} semaines | Test : {len(X_test)} semaines\n")
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
        #on lui passe des hyperparametres de qualité
        model = initialize_model(n_estimators= 1000,learning_rate= 0.1,max_depth=2,subsample=0.8,colsample_bytree =0.6, min_child_weight=5,random_state = 42)

        #model = initialize_model()

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
# 6. PREDICT — prévision 13 prochaines semaines
# ══════════════════════════════════════════════════════════════════════════════


def pred(model, df_ml: pd.DataFrame) -> pd.Series:
    """
    Génère les prévisions sur toute la période de test (Mars → Mai 2026),
    soit environ 12 à 13 semaines selon le calendrier.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: pred" + Style.RESET_ALL)

    # 1. On filtre df_ml pour ne garder QUE la période cible (Mars à Mai 2026)
    # C'est exactement les mêmes dates que ton X_test
    df_futur = df_ml.loc[params.EVALUATION_START_DATE:TEST_END]

    # 2. On extrait les features matricielles (X) pour cette période
    X_future = df_futur[FEATURE_COLS].values

    # 3. XGBoost génère autant de prédictions qu'il y a de lignes (12 ou 13)
    y_pred = predict_model(model, X_future)

    # 4. On crée la Series Pandas en utilisant directement l'index temporel réel de cette période
    forecast = pd.Series(y_pred, index=df_futur.index, name=params.TARGET_COL)

    print(f"\n✅ pred() done — Predicted {len(forecast)} weeks:\n{forecast.to_string()}\n")
    return forecast


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — données chargées une seule fois, modèle partagé
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_time = time.time()

    # load from big query
    #df = clean_piezo(load_piezo_bq(DATA_CODE_PIEZO))
    df = clean_piezo2(load_piezo_bq(DATA_CODE_PIEZO))

    df_ml = preprocess_week(df)

    X_train, X_test, y_train, y_test = split_data(df_ml)

    # 2. Train — optimize=True pour GridSearchCV
    #model, history = train(X_train, y_train, optimize=True)
    model, history = train(X_train, y_train, optimize=False)

    # 3. Evaluate — même modèle, pas de rechargement
    metrics = evaluate(model, X_test, y_test)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n" + "="*45)
    print(Fore.GREEN + f"⏱️  Temps total d'exécution : {execution_time:.2f} secondes" + Style.RESET_ALL)
    print("="*45)

import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model
from hydrosense.ml_logic.folding import get_folds

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.preprocess.preprocessor import preprocess_week

from hydrosense import params

# ══════════════════════════════════════════════════════════════════════════════
# PARAMS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = Path("data/piezo_bourdet_clean.csv")
DATA_CODE_PIEZO = "BSS001QHYH"
TARGET_COL   = "niveau_nappe_eau"
DATE_COL     = "date_mesure"

# A revoir
FEATURE_COLS = ["mois", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]

# Split : 3 derniers mois en test (Mars → Mai 2026)
# EN DUR --- OUILLE OUILLE

TRAIN_END  = "2025-02-28"
TEST_START = "2025-03-01"
TEST_END   = "2025-05-31"


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
    y = df_ml[params.TARGET_COL]

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
            # date_column_name=DATE_COL,  # Plus nécessaire car optimize_model ne le passe plus à get_folds
            n_splits_cv=3,
            val_months_duration=3 # Durée de la validation en mois pour la CV
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

    # 1. Données — une seule fois
    #df    = load_data() # du CSV
    # load from big query
    df = clean_piezo(load_piezo_bq(DATA_CODE_PIEZO))

    df_ml = preprocess_week(df)  # OR preprocess_week ???
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
            val_months_duration=3
        )
        for i, (train_idx, val_idx) in enumerate(test_splits):
            # Utilise X_train_df pour récupérer les dates réelles pour la visualisation
            train_period = X_train_df.iloc[train_idx].index
            val_period = X_train_df.iloc[val_idx].index
            print(f"  Split {i+1}:")
            print(f"    Train: {train_period.min().strftime('%Y-%m')} à {train_period.max().strftime('%Y-%m')} ({len(train_idx)} samples)")
            print(f"    Val:   {val_period.min().strftime('%Y-%m')} à {val_period.max().strftime('%Y-%m')} ({len(val_idx)} samples)")
            # Assertions pour vérifier l'absence de fuite temporelle et l'ordre chronologique
            assert train_period.max() < val_period.min(), f"Fuite temporelle détectée dans Split {i+1}!" # Vérifie l'ordre chronologique
            assert len(val_period) == 3, f"La durée de validation n'est pas respectée dans Split {i+1}!" # Vérifie le nombre exact de mois
    except ValueError as e:
        print(f"  Erreur lors de la visualisation des splits: {e}")
    print(Fore.CYAN + "--- Fin de la visualisation des splits ---" + Style.RESET_ALL)

    # 2. Train — optimize=True pour GridSearchCV
    model, history = train(X_train_df, y_train_df, optimize=True)

    # 3. Evaluate — même modèle, pas de rechargement
    metrics = evaluate(model, X_test_df.values, y_test_df.values)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)


# probleme de dataleakage ??? 
def pred_future(model, df_ml: pd.DataFrame, n_weeks: int = 13) -> pd.Series:
    """
    Génère les prévisions sur les n_weeks prochaines semaines (futur réel).
    Utilise une boucle autorégressive — chaque semaine prédit la suivante.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: pred_future" + Style.RESET_ALL)

    FEATURE_COLS = ["semaine", "lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "moyenne_3w", "moyenne_6w"]
    df_future = df_ml.copy()
    predictions = []

    for i in range(n_weeks):
        last_row = df_future[FEATURE_COLS].tail(1).values
        y_next = predict_model(model, last_row)[0]
        next_date = df_future.index[-1] + pd.Timedelta(weeks=1)

        new_row = pd.DataFrame({
            "niveau_nappe_eau": [y_next],
            "semaine":    [next_date.isocalendar().week],
            "lag_1":      [y_next],
            "lag_2":      [df_future["niveau_nappe_eau"].iloc[-1]],
            "lag_3":      [df_future["niveau_nappe_eau"].iloc[-2]],
            "lag_4":      [df_future["niveau_nappe_eau"].iloc[-3]],
            "lag_52":     [df_future["niveau_nappe_eau"].iloc[-51]],
            "moyenne_3w": [df_future["niveau_nappe_eau"].tail(3).mean()],
            "moyenne_6w": [df_future["niveau_nappe_eau"].tail(6).mean()],
        }, index=[next_date])

        df_future = pd.concat([df_future, new_row])
        predictions.append((next_date, round(float(y_next), 3)))

    forecast = pd.Series(
        [p[1] for p in predictions],
        index=[p[0] for p in predictions],
        name=params.TARGET_COL
    )

    print(f"\n✅ pred_future() done — {len(forecast)} semaines prédites\n")
    return forecast
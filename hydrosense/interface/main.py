import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from typing import Tuple

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model
from hydrosense.ml_logic.folding import get_folds

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.preprocess.preprocessor import preprocess_week


from hydrosense import params

# ══════════════════════════════════════════════════════════════════════════════
#  TODO : GERER PARAMS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_TYPE = 'XGB'  # ['LINEAR','BAG'] ??

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
    Évalue le modèle sur les (3 mois) de test (jamais vus à l'entraînement).

    Returns
    -------
    metrics : dict {"mae": float, "rmse": float, "r2": float}
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    if False:
        print('Evaluate en on folds.')

    metrics = evaluate_model(model, X_test, y_test)
    # evaluate deeper ?

    print("✅ evaluate() done \n")
    return metrics

def evaluate_deeper(X_df: pd.DataFrame, y_df: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-validation evaluation of the model using a time-series aware
    folding strategy (get_folds).
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate_deeper (Cross-Validation)" + Style.RESET_ALL)


    splits = get_folds(
        dates_series=X_df.index,
        n_splits=5,
        min_train_years=3,
        val_months_duration=3
    )

    metrics_per_fold = []
    all_predictions = pd.DataFrame()

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {i+1}/{len(splits)} ---")

        X_train_fold = X_df.iloc[train_idx]
        y_train_fold = y_df.iloc[train_idx]
        X_val_fold = X_df.iloc[val_idx]
        y_val_fold = y_df.iloc[val_idx]


        model_fold = initialize_model()
        model_fold, _ = train_model(model_fold, X_train_fold.values, y_train_fold.values)

        # # 4. Predict and evaluate on train and validation sets
        # y_pred_val = pd.Series(model_fold.predict(X_val_fold.values), index=y_val_fold.index)
        # y_pred_train = pd.Series(model_fold.predict(X_train_fold.values), index=y_train_fold.index)

        metrics_val = evaluate_model(model_fold, X_val_fold.values, y_val_fold.values)
        metrics_train = evaluate_model(model_fold, X_train_fold.values, y_train_fold.values)

        print(f"  Train R²: {metrics_train['r2']:.3f} | Val R²: {metrics_val['r2']:.3f}")
        print(f"  Train MAE: {metrics_train['mae']:.3f} | Val MAE: {metrics_val['mae']:.3f}")

        metrics_per_fold.append({
            'fold': i + 1, 'val_start': y_val_fold.index.min(), 'val_end': y_val_fold.index.max(),
            'r2_train': metrics_train['r2'], 'mae_train': metrics_train['mae'], 'rmse_train': metrics_train['rmse'], 'max_error_train': metrics_train['max_error'],
            'r2_val': metrics_val['r2'], 'mae_val': metrics_val['mae'], 'rmse_val': metrics_val['rmse'], 'max_error_val': metrics_val['max_error']
        })

        # # Store predictions for visualization
        # fold_predictions = pd.DataFrame({'date': y_val_fold.index, 'actual': y_val_fold, 'forecast': y_pred_val, 'fold': i + 1})
        # all_predictions = pd.concat([all_predictions, fold_predictions])

    metrics_df = pd.DataFrame(metrics_per_fold)
    print("\n--- Cross-Validation Summary ---")
    print("Average metrics on validation sets:")
    print(metrics_df[['r2_val', 'mae_val', 'rmse_val', 'max_error_val']].mean().round(3))

    print("\n✅ evaluate_deeper() done \n")
    return metrics_df, all_predictions

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

    df_ml = preprocess(df)  # OR preprocess_week ???
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

    # 3b. Deeper Evaluation with Cross-Validation on the training set
    print(Fore.CYAN + "\n--- Évaluation approfondie par Cross-Validation ---" + Style.RESET_ALL)
    cv_metrics, cv_predictions = evaluate_deeper(X_train_df, y_train_df)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

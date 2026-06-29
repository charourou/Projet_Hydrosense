import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from typing import Tuple

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
from sklearn.linear_model import Lasso

from hydrosense.ml_logic.model import initialize_model, optimize_model, train_model, evaluate_model, predict_model
from hydrosense.ml_logic.folding import get_folds
from hydrosense.ml_logic.base import BaselineLastYear

from hydrosense.database.bigquery import load_piezo_bq, load_plean
from hydrosense.preprocess.cleaning import clean_piezo, clean_piezo2
from hydrosense.preprocess.preprocessor import preprocess_week, split_data


from hydrosense import params


# ══════════════════════════════════════════════════════════════════════════════
#  TODO : GERER PARAMS
# ══════════════════════════════════════════════════════════════════════════════

MODEL_TYPE = 'LASSO'  # ['LASSO', 'BASE', 'XGB']
DATA_PATH    = Path("data/piezo_bourdet_clean.csv")
DATA_CODE_PIEZO = "BSS001QHYH"
DATE_COL     = "date_mesure"

FEATURE_COLS = [
    # Saisonnalité cyclique pure
    "semaine_sin", "semaine_cos",

    # Historique de la nappe (Target)
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_52",

    # Uniquement le lag 1 pour les PC
    "PC1_lag_1",
    "PC2_lag_1",
    "PC3_lag_1",

    # Lags 1 à 4 pour PU_synth
    "PU_synth_lag_1", "PU_synth_lag_2", "PU_synth_lag_3", "PU_synth_lag_4"
]
FEATURE_COLS = ["semaine_sin","semaine_cos", "lag_1", "lag_2", "lag_3","lag_4" ,"lag_52", "moyenne_3w", "moyenne_6w","RR_synth"]
FEATURE_COLS = ["semaine_sin","semaine_cos", "PU_synth", "PC1", "PC2", "PC3"]


# Split : 3 derniers mois en test (Mars → Mai 2025)
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
    - Rééchantillonne les données journalières en moyennes MENSUELLES
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

    df_ml["lag_1"]      = df_ml[params.TARGET_COL].shift(1)
    df_ml["lag_2"]      = df_ml[params.TARGET_COL].shift(2)
    df_ml["lag_3"]      = df_ml[params.TARGET_COL].shift(3)
    df_ml["lag_12"]     = df_ml[params.TARGET_COL].shift(12)
    df_ml["moyenne_3m"] = df_ml[params.TARGET_COL].rolling(window=3).mean()
    df_ml["moyenne_6m"] = df_ml[params.TARGET_COL].rolling(window=6).mean()
    df_ml = df_ml.dropna()

    print(f"✅ preprocess() done — {len(df_ml)} mois | {df_ml.shape[1]} colonnes\n")
    return df_ml


def preprocess_slim(df: pd.DataFrame) -> pd.DataFrame:
    """ DONNEES MENSUELLES"""

    y_mensuel = df[params.TARGET_COL].resample("ME").mean()

    # Feature engineering
    df_ml = pd.DataFrame(y_mensuel)

    df_ml["mois_sin"] = np.sin(2 * np.pi * df_ml.index.month / 12)
    df_ml["mois_cos"] = np.cos(2 * np.pi * df_ml.index.month / 12)

    df_ml["lag_1"]      = df_ml[params.TARGET_COL].shift(1)
    df_ml["lag_2"]      = df_ml[params.TARGET_COL].shift(2)
    df_ml["lag_3"]      = df_ml[params.TARGET_COL].shift(3)
    df_ml["lag_12"]     = df_ml[params.TARGET_COL].shift(12)


    print(f"Preprocess SLIM done — {len(df_ml)} mois | {df_ml.shape[1]} colonnes\n")
    return df_ml



def train(X_train_df: pd.DataFrame, y_train_df: pd.Series,
          pick_model = None,
          optimize = False):
    """
    X_train, y_train : données d'entraînement
    optimize         : True → GridSearchCV | False → hyperparamètres par défaut

    Returns
    -------
    (model, history)
    """

    if not pick_model: # local override of the model
        pick_model = MODEL_TYPE


# ════════════ CAS 1 : BASELINE NAÏVE ════════════
    if pick_model == 'BASE':
        print("⏳ Initialisation de la persistance annuelle (J-365)...")
        model = BaselineLastYear()
        model.fit(X_train_df, y_train_df)
        print("✅ train() done — Baseline prête.")
        history = {}
        return model, history

# =============================================================================================
    if pick_model == 'LASSO':
        print(Fore.MAGENTA + "\nUse case: train (Mode Baseline Lasso)" + Style.RESET_ALL)
        model = Lasso(alpha=0.01, random_state=42)
        model.fit(X_train_df.values, y_train_df.values)

        history = {}
        print("✅ train() done — Modèle Lasso entraîné.\n")
        return model, history

    print(Fore.MAGENTA + "\nUse case: train" + Style.RESET_ALL)

# =============================================================================================
    if pick_model == 'XGB':
        print(Fore.MAGENTA + "\n⭐️ Use case: train (Mode XGB)" + Style.RESET_ALL)
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
            #on lui passe des hyperparametres de qualité
            model = initialize_model(n_estimators= 1000,
                                    learning_rate= 0.1,
                                    max_depth=2,
                                    subsample=0.8,
                                    colsample_bytree =0.6,
                                    min_child_weight=5,
                                    random_state = 42)

        model, history = train_model(model, X_train_df.values, y_train_df.values)

        print("✅ train() done \n")
        return model, history


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

    metrics = evaluate_model(model, X_test, y_test)
    # evaluate deeper ?

    print("✅ evaluate() done \n")
    return metrics

def evaluate_deeper(X_df: pd.DataFrame, y_df: pd.Series, splits,
                    X_test_df: pd.DataFrame, y_test_df: pd.Series
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a cross-validation evaluation of the model using a time-series aware
    folding strategy (get_folds).
    """
    print(Fore.MAGENTA + "\n⭐️ Evaluate_deeper (Cross-Validation)" + Style.RESET_ALL)

    metrics_per_fold = []
    all_predictions = pd.DataFrame()

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {i+1}/{len(splits)} ---")

        print(f"Debug: Current fold {i+1}")
        print(f"Debug: len(X_df) = {len(X_df)}")
        print(f"Debug: len(train_idx) = {len(train_idx)}")
        if train_idx:
            print(f"Debug: max(train_idx) = {max(train_idx)}")


        X_train_fold = X_df.iloc[train_idx]
        y_train_fold = y_df.iloc[train_idx]
        X_val_fold = X_df.iloc[val_idx]
        y_val_fold = y_df.iloc[val_idx]

        # Train model on this fold
        model_fold = initialize_model()
        model_fold, _ = train_model(model_fold, X_train_fold, y_train_fold)

        # Evaluate on Train, Validation (fold) and Test (unseen)
        metrics_train = evaluate_model(model_fold, X_train_fold, y_train_fold,
                                       y_train=y_train_fold,
                                       verbose=False)
        metrics_val = evaluate_model(model_fold, X_val_fold, y_val_fold,
                                     y_train=y_train_fold,
                                     verbose=False)

        # Global test set evaluation (X_test_df/y_test_df are defined in global scope or passed)
        # Here we assume we want to see how a model trained on a subset performs on the final test set
        metrics_test = evaluate_model(model_fold, X_test_df, y_test_df,
                                      y_train=y_train_fold, verbose=False)

        print(f"  Fold {i+1} | Train MAE: {metrics_train['mae']:.3f} | Val MAE: {metrics_val['mae']:.3f} | Test MAE: {metrics_test['mae']:.3f}")
        print(f"  Fold {i+1} | Train RMSE: {metrics_train['rmse']:.3f} | Val RMSE: {metrics_val['rmse']:.3f} | Test RMSE: {metrics_test['rmse']:.3f}")
        print(f"  Fold {i+1} | Train RMSSE: {metrics_train['rmsse'] if np.isfinite(metrics_train['rmsse']) else 'inf'} | Val RMSSE: {metrics_val['rmsse'] if np.isfinite(metrics_val['rmsse']) else 'inf'} | Test RMSSE: {metrics_test['rmsse'] if np.isfinite(metrics_test['rmsse']) else 'inf'}")


        metrics_per_fold.append({
            'fold': i + 1,
            'train_size': len(train_idx),
            'mae_train': metrics_train['mae'], 'rmse_train': metrics_train['rmse'], 'r2_train': metrics_train['r2'], 'max_error_train': metrics_train['max_error'], 'rmsse_train': metrics_train['rmsse'],
            'mae_val': metrics_val['mae'], 'rmse_val': metrics_val['rmse'], 'r2_val': metrics_val['r2'], 'max_error_val': metrics_val['max_error'], 'rmsse_val': metrics_val['rmsse'],
            'mae_test': metrics_test['mae'], 'rmse_test': metrics_test['rmse'], 'r2_test': metrics_test['r2'], 'max_error_test': metrics_test['max_error'], 'rmsse_test': metrics_test['rmsse']
        })

    metrics_df = pd.DataFrame(metrics_per_fold)
    print("\n--- Learning Curve / CV Summary ---")
    print("Average metrics on training sets:")
    print(metrics_df[['mae_train', 'rmse_train', 'r2_train', 'max_error_train', 'rmsse_train']].mean().round(3))
    print("\nAverage metrics on validation sets:")
    print(metrics_df[['mae_val', 'rmse_val', 'r2_val', 'max_error_val', 'rmsse_val']].mean().round(3))
    print("\nAverage metrics on test sets (from models trained on CV folds):")
    print(metrics_df[['mae_test', 'rmse_test', 'r2_test', 'max_error_test', 'rmsse_test']].mean().round(3))
    print("\n✅ evaluate_deeper() done \n")
    return metrics_df, all_predictions

# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICT — prévision sur 13 prochaines semaines
# ══════════════════════════════════════════════════════════════════════════════


def pred(model, df_ml: pd.DataFrame) -> pd.Series:
    """
    Génère les prévisions sur toute la période de test ( Printemps ),
    soit environ 12 à 13 semaines selon le calendrier.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: pred" + Style.RESET_ALL)

    # On filtre df_ml pour ne garder QUE la période cible (Printemps)
    df_futur = df_ml.loc[params.EVALUATION_START_DATE:TEST_END]

    # On extrait les features
    X_future = df_futur[FEATURE_COLS].values

    # Le modèle génère autant de prédictions qu'il y a de lignes
    # WARNING : ceci est une prevision LIEE
    # DATA LEAKAGE
    y_pred = predict_model(model, X_future)

    # On crée la Series Pandas en utilisant directement l'index temporel réel de cette période
    forecast = pd.Series(y_pred,
                         index=df_futur.index,
                         name=params.TARGET_COL)

    print(f"\n✅ pred() done — Predicted {len(forecast)} weeks:\n{forecast.to_string()}\n")
    return forecast


def pred_future(model, df_ml: pd.DataFrame, n_weeks: int = 13) -> pd.Series:
    """
    Génère les prévisions sur les n_weeks prochaines semaines (futur réel).
    Utilise une boucle autorégressive — chaque semaine prédit la suivante.
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: pred_future" + Style.RESET_ALL)

    df_future = df_ml.copy()
    predictions = []

    # Scenario de pluie.
    SECHERESSE = np.min(df_future["PU_synth_lag_1"])
    SECHERESSE = -1


    # TODO : check hard ou soft list <<<<<<<< ??
    # VERSION LITE - sans ['PC1_lag_1',  'PC2_lag_1', 'PC3_lag_1' ]
    TRAINING_FEATURES =  ['semaine_sin', 'semaine_cos',
                            'niveau_nappe_eau_lag_1', 'niveau_nappe_eau_lag_2', 'niveau_nappe_eau_lag_3',
                            'niveau_nappe_eau_lag_4',
                            'niveau_nappe_eau_lag_52',
                            'PU_synth_lag_1', 'PU_synth_lag_2', 'PU_synth_lag_3', 'PU_synth_lag_4'
                            ]



    for i in range(n_weeks):

        # prevision à date t (lag 0)
        last_row = df_future[TRAINING_FEATURES].tail(1).values
        y_next = predict_model(model, last_row)[0]



        next_date = df_future.index[-1] + pd.Timedelta(weeks=1)
        semaine = next_date.isocalendar().week

        # ATTENTION RIGIDE
        # creer un fonction pour mettre en place la nouvelle ligne
        new_row = pd.DataFrame({
            "semaine_sin":    [np.sin(2 * np.pi * semaine / 52)]   ,
            "semaine_cos":    [np.cos(2 * np.pi * semaine / 52)]   ,
            "niveau_nappe_eau_lag_1":      [y_next],
            "niveau_nappe_eau_lag_2":      [df_future["niveau_nappe_eau_lag_1"].iloc[-1]],
            "niveau_nappe_eau_lag_3":      [df_future["niveau_nappe_eau_lag_2"].iloc[-1]],
            "niveau_nappe_eau_lag_4":      [df_future["niveau_nappe_eau_lag_3"].iloc[-1]],
            "niveau_nappe_eau_lag_52":     [df_future["niveau_nappe_eau_lag_1"].iloc[-50]],

            "PU_synth_lag_1" :[SECHERESSE],  # Hypothese SECHERESSE -----
            "PU_synth_lag_2" :[df_future["PU_synth_lag_1"].iloc[-1]],
            "PU_synth_lag_3" :[df_future["PU_synth_lag_2"].iloc[-1]],
            "PU_synth_lag_4" :[df_future["PU_synth_lag_3"].iloc[-1]],
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


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — données chargées une seule fois, modèle partagé
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # load from big query
    #df = clean_piezo(load_piezo_bq(DATA_CODE_PIEZO))
    df = load_plean(DATA_CODE_PIEZO)

    if params.DATE_COL in df.columns:
        df[params.DATE_COL] = pd.to_datetime(df[params.DATE_COL])
        df.set_index(params.DATE_COL, inplace=True)
    df = df.sort_index()

    df_ml = preprocess_week(df)  # OR preprocess_week ???

    X_train_df, X_test_df, y_train_df, y_test_df = split_data(
        df_ml,
        feature_cols=FEATURE_COLS,
        target_col=params.TARGET_COL,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END
    )

    # --- Zone de test pour visualiser les splits de cross-validation ---
    print(Fore.CYAN + "\n--- Visualisation des splits de cross-validation annuels ---" + Style.RESET_ALL)
    try:
        # get_folds attend un DataFrame avec la colonne de dates.
        test_splits = get_folds(
            # Passe directement le DatetimeIndex de X_train_df
            dates_series=X_train_df.index,
            n_splits= 20 , # Nombre de splits à visualiser
            min_train_years=3,
            val_months_duration=3
        )
        for i, (train_idx, val_idx) in enumerate(test_splits):

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
    model, history = train(X_train_df, y_train_df, optimize=False)

    # 3. Evaluate — même modèle, pas de rechargement
    metrics = evaluate(model, X_test_df, y_test_df)
    print(metrics)

    # 3b. Deeper Evaluation with Cross-Validation on the training set and final test set
    print(Fore.CYAN + "\n--- Évaluation approfondie par Cross-Validation ---" + Style.RESET_ALL)
    cv_metrics, cv_predictions = evaluate_deeper(X_train_df, y_train_df, test_splits, X_test_df, y_test_df)
    print(cv_predictions)
    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

    # 4. Predict — prévision 3 mois futurs
    forecast = pred(model, df_ml)

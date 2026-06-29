import numpy as np
import pandas as pd
import time
from colorama import Fore, Style
from typing import Tuple, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from hydrosense.ml_logic.folding import get_folds


# Timing the XGBoost import
print(Fore.BLUE + "\nLoading XGBoost..." + Style.RESET_ALL)
start = time.perf_counter()
from xgboost import XGBRegressor
end = time.perf_counter()
print(f"\n✅ XGBoost loaded ({round(end - start, 2)}s)")


def initialize_model(
    n_estimators: int = 1000,
    learning_rate: float = 0.1,
    max_depth: int = 2,
    subsample: float = 0.8,
    colsample_bytree: float = 0.6,
    min_child_weight=5,
    random_state: int = 42
) -> XGBRegressor:
    """
    Initialize the XGBoost Regressor with given hyperparameters.

    Parameters
    ----------
    n_estimators     : number of boosting rounds (trees)
    learning_rate    : step size shrinkage to prevent overfitting
    max_depth        : maximum depth of each tree
    subsample        : fraction of samples used per tree
    colsample_bytree : fraction of features used per tree
    random_state     : reproducibility seed

    Returns
    -------
    XGBRegressor (not yet fitted)
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight = min_child_weight,
        objective="reg:squarederror",   # régression → MSE comme loss
        random_state=random_state,
        n_jobs=-1,                       # utilise tous les cœurs CPU
        verbosity=0
    )

    print("Model initialized")
    return model


def optimize_model(
    X: np.ndarray,
    y: np.ndarray,
    X_train_df_for_folds: Optional[pd.DataFrame] = None, # Le DataFrame complet pour les folds
    # date_column_name: str = 'date_mesure', # Plus nécessaire car get_folds prend un DatetimeIndex
    n_splits_cv: int = 3, # Nombre de splits pour la CV
    val_months_duration: int = 3, # Durée de la validation en mois
    min_train_years: int = 3 # Nombre minimum d'années pour le premier jeu d'entraînement
) -> Tuple[XGBRegressor, dict]:
    """
    Run a GridSearchCV to find the best XGBoost hyperparameters.
    Exécute un GridSearchCV pour trouver les meilleurs hyperparamètres XGBoost
    en utilisant une stratégie de cross-validation chronologique annuelle.

    Parameters
    ----------
    X  : feature matrix (lags, mois, moyennes mobiles…)
    y  : target — niveau de la nappe (float)
    cv : number of cross-validation folds

    Returns
    -------
    (best_model, best_params)
    """
    print(Fore.BLUE + "\nOptimizing hyperparameters..." + Style.RESET_ALL)

    param_grid = {
        "n_estimators":     [100, 300, 500],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [2, 3, 4],
        "subsample":        [0.6, 0.7, 0.8],
        "colsample_bytree": [0.5, 0.6, 0.7],
        # 💡 ON AJOUTE LE POIDS MINIMUM PAR FEUILLE (crucial pour le bruit hebdo)
         "min_child_weight": [3, 5, 10]
    }

    # Définition de la stratégie de découpage temporel
    if X_train_df_for_folds is not None:
        # Utilise la stratégie de fenêtre expansive annuelle
        cv_strategy = get_folds(
            # Passe directement le DatetimeIndex du DataFrame d'entraînement
            dates_series=X_train_df_for_folds.index,
            n_splits=n_splits_cv,
            min_train_years=min_train_years,
            val_months_duration=val_months_duration
        )
    else:
        # Fallback au TimeSeriesSplit standard si les dates ne sont pas fournies
        print(Fore.YELLOW + "Warning: dates_train not provided. Falling back to TimeSeriesSplit." + Style.RESET_ALL)
        cv_strategy = TimeSeriesSplit(n_splits=n_splits_cv)

    grid_search = GridSearchCV(
        estimator=XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0),
        param_grid=param_grid,
        cv=cv_strategy,                     # Utilise la stratégie de CV temporelle
        scoring="neg_mean_squared_error",   # Ou 'neg_mean_absolute_error' selon la préférence
        n_jobs=-1,                          # Utilise tous les cœurs disponibles
        verbose=1                           # Affiche la progression
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_model  = grid_search.best_estimator_

    print(f"✅ Optimization complete. Best params: {best_params}")
    return best_model, best_params

def train_model(
    model: XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    early_stopping_rounds: int = 20
) -> Tuple[XGBRegressor, dict]:
    """
    Fit the XGBoost model. Supports early stopping if validation data is provided.

    Parameters
    ----------
    model                 : initialized XGBRegressor
    X, y                  : training data
    X_val, y_val          : optional validation set for early stopping
    early_stopping_rounds : stop if no improvement after N rounds

    Returns
    -------
    (fitted_model, training_info dict)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)
    start = time.perf_counter()

    fit_params = {}
    if X_val is not None and y_val is not None:
        # Early stopping : arrête l'entraînement si le val_loss stagne
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"]  = False

    model.fit(X, y, **fit_params)

    end = time.perf_counter()

    # Calcul de l'erreur sur le train pour le résumé
    y_pred_train = model.predict(X)
    train_mae    = mean_absolute_error(y, y_pred_train)
    train_rmse   = np.sqrt(mean_squared_error(y, y_pred_train))

    history = {
        "train_mae":        round(train_mae, 4),
        "train_rmse":       round(train_rmse, 4),
        "n_estimators_used": model.best_iteration + 1 if hasattr(model, "best_iteration") and model.best_iteration else model.n_estimators,
        "training_time_s":  round(end - start, 2)
    }

    print(f"✅ Model trained on {len(X)} rows in {history['training_time_s']}s")
    print(f"   Train MAE  : {history['train_mae']}")
    print(f"   Train RMSE : {history['train_rmse']}")

    return model, history

def evaluate_model(
    model: XGBRegressor,
    X: np.ndarray,
    y: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    verbose = True
        ) -> dict:
    """
    Evaluate the trained model on a given set.

    Parameters
    ----------
    model : fitted XGBRegressor
    X, y  : test data (jamais vus pendant l'entraînement)
    y_train: optionel pour calcul RMSSE

    Returns
    -------
    metrics dict : mae, rmse, r2, max_error
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    r2   = r2_score(y, y_pred)
    max_err = max([0,
                   float(max(y-y_pred))]
                   )

    # RMSSE calculation
    rmsse = np.nan
    if y_train is not None and len(y_train) > 1:
        # Denominator is the RMSE of a naive one-step forecast on the training data.
        naive_mse_on_train = mean_squared_error(y_train[1:], y_train[:-1])
        rmse_naive_on_train = np.sqrt(naive_mse_on_train)
        if rmse_naive_on_train > 1e-9:  # Avoid division by zero
            rmsse = rmse / rmse_naive_on_train
        else:
            rmsse = np.inf if rmse > 1e-9 else 0.0
        # Naive model is perfect, if our model is also perfect, RMSSE is 0

    metrics = {
        "mae":  round(mae, 4),
        "rmse": round(rmse, 4),
        "r2":   round(r2, 4),
        "max_error": round(max_err, 4),
        "rmsse" : round(rmsse, 4) if np.isfinite(rmsse) else rmsse
            }
    if verbose:
        print(f"✅ Model evaluated on set (train, val ou test)")
        print(f"   MAE  : {metrics['mae']}  (erreur moyenne en mètres NGF)")
        print(f"   RMSE : {metrics['rmse']} (pénalise les grandes erreurs)")
        print(f"   R²   : {metrics['r2']}  (1.0 = parfait)")
        print(f"   Max Error: {metrics['max_error']} (erreur maximale absolue)")
        print(f"   RMSSE: {metrics['rmsse']} (erreur par rapport au choix naif)")

    return metrics

def predict_model(model, X) -> np.ndarray:
    """    Generate predictions for new input data.    """
    if model is None:
        print(f"\n❌ No model to predict with")
        return None

    y_pred = model.predict(X)
    # print(f"✅ Predicted {len(y_pred)} values")
    return y_pred

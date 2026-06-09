import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from hydrosense.ml_logic.folding import get_folds
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.interface.main import preprocess, TARGET_COL # Importe les constantes et la fonction preprocess

def arima_cross_validate(
    dates_series: pd.DatetimeIndex,
    y_series: pd.Series,
    arima_order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 12),
    n_splits_cv: int = 5,
    min_train_years: int = 3,
    val_months_duration: int = 3
    val_duration_value: int = 3, # Renamed from val_months_duration
    freq: str = 'M' # Added frequency parameter
) -> dict:
    """
    Effectue une cross-validation pour un modèle ARIMA en utilisant la stratégie
    de découpage annuel expansif.

    Args:
        dates_series (pd.DatetimeIndex): L'index temporel correspondant à la série chronologique.
        y_series (pd.Series): La série chronologique à modéliser.
        arima_order (tuple): L'ordre (p, d, q) du modèle ARIMA.
        seasonal_order (tuple): L'ordre saisonnier (P, D, Q, S) du modèle ARIMA.
        n_splits_cv (int): Nombre de splits pour la cross-validation.
        min_train_years (int): Nombre minimum d'années pour le premier jeu d'entraînement.
        val_months_duration (int): Durée du jeu de validation en mois.
        val_duration_value (int): Duration of the validation set. Its unit depends on `freq`.
        freq (str): Frequency of the data. 'M' for monthly, 'W' for weekly.

    Returns:
        dict: Un dictionnaire contenant les métriques d'évaluation pour chaque fold et la moyenne.
    """
    print(f"Démarrage de la cross-validation ARIMA avec {n_splits_cv} splits...")

    all_metrics = []

    try:
        splits = get_folds(
            dates_series=dates_series,
            n_splits=n_splits_cv,
            min_train_years=min_train_years,
            val_months_duration=val_months_duration
            val_duration_value=val_duration_value, # Renamed
            freq=freq # Added
        )
    except ValueError as e:
        print(f"Erreur lors de la génération des folds: {e}")
        return {}

    if not splits:
        print("Aucun split généré pour la cross-validation.")
        return {}

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {i+1}/{len(splits)} ---")

        y_train_fold = y_series.iloc[train_idx]
        y_val_fold = y_series.iloc[val_idx]

        print(f"  Période d'entraînement: {y_train_fold.index.min().year}-{y_train_fold.index.max().year} ({len(y_train_fold)} échantillons)")
        print(f"  Période de validation: {y_val_fold.index.min().year}-{y_val_fold.index.max().year} ({len(y_val_fold)} échantillons)")

        if len(y_train_fold) == 0 or len(y_val_fold) == 0:
            print("  Fold ignoré en raison d'un ensemble d'entraînement ou de validation vide.")
            continue

        try:
            # Initialisation et ajustement du modèle SARIMAX
            model = SARIMAX(
                y_train_fold,
                order=arima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            arima_fitted = model.fit(disp=False) # disp=False pour supprimer la sortie verbeuse

            # Prévision pour la période de validation
            forecast = arima_fitted.predict(start=y_val_fold.index[0], end=y_val_fold.index[-1])

            # Assurer que la prévision et les valeurs réelles ont le même index pour le calcul des métriques
            forecast = forecast.reindex(y_val_fold.index)

            # Évaluation
            mae = mean_absolute_error(y_val_fold, forecast)
            rmse = np.sqrt(mean_squared_error(y_val_fold, forecast))

            print(f"  MAE du fold: {mae:.4f}")
            print(f"  RMSE du fold: {rmse:.4f}")
            all_metrics.append({"fold": i + 1, "mae": mae, "rmse": rmse})

        except Exception as e:
            print(f"  Erreur lors de l'ajustement ou de la prévision ARIMA pour le fold {i+1}: {e}")
            all_metrics.append({"fold": i + 1, "mae": np.nan, "rmse": np.nan})

    if all_metrics:
        avg_mae = np.nanmean([m["mae"] for m in all_metrics])
        avg_rmse = np.nanmean([m["rmse"] for m in all_metrics])
        print(f"\n--- Cross-validation terminée ---")
        print(f"MAE moyenne sur {len(all_metrics)} folds: {avg_mae:.4f}")
        print(f"RMSE moyenne sur {len(all_metrics)} folds: {avg_rmse:.4f}")
        return {"all_folds": all_metrics, "average_mae": avg_mae, "average_rmse": avg_rmse}
    else:
        print("Aucune métrique collectée.")
        return {}

if __name__ == "__main__":
    # --- Chargement et prétraitement des données (similaire à main.py) ---
    print("Chargement des données pour la démo de cross-validation ARIMA...")
    DATA_CODE_PIEZO = "BSS001QHYH" # ID de piézomètre exemple
    df_raw = load_piezo_bq(DATA_CODE_PIEZO)
    df_clean = clean_piezo(df_raw)
    df_ml = preprocess(df_clean)

    # Extraire la série cible et son DatetimeIndex
    y_series_for_arima = df_ml[TARGET_COL]
    dates_series_for_arima = y_series_for_arima.index

    # --- Exécuter la cross-validation ARIMA ---
    results = arima_cross_validate(
        dates_series=dates_series_for_arima,
        y_series=y_series_for_arima,
        arima_order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        n_splits_cv=5,
        min_train_years=3,
        val_months_duration=3
        val_duration_value=3, # Renamed
        freq='M' # Assuming monthly data from preprocess()
    )

    print("\nRésultats de la Cross-Validation ARIMA:")
    print(results)

import random
import pandas as pd
import numpy as np

def get_folds(dates_series: pd.DatetimeIndex, n_splits: int = 5, min_train_years: int = 3, val_years_duration: int = 1):
    """
    Génère des indices de train/validation pour une cross-validation temporelle
    en utilisant une stratégie de fenêtre expansive annuelle. Le jeu d'entraînement commence toujours au début des données et s'étend
    année après année. Le jeu de validation est l'année (ou les années)
    immédiatement suivante(s) au jeu d'entraînement.

    Exemple avec min_train_years=3, val_years_duration=1:
    Si les dates vont de 2010 à 2020 et n_splits=3:
    Split 1: Train: 2010-2015, val: 2016 (assuming enough data for 3 splits)
    Split 2: Train: 2010-2016, val: 2017
    Split 3: Train: 2010-2017, val: 2018

    Arguments:
        dates_series (pd.DatetimeIndex): L'index temporel des données (X_train).
        n_splits (int): Le nombre de splits à générer (similaire à TimeSeriesSplit).
        min_train_years (int): Nombre minimum d'années pour le premier jeu d'entraînement.
                               Le premier split aura au moins `min_train_years` dans son train set.
        val_years_duration (int): Nombre d'années à utiliser pour chaque jeu de val.

    Retourne:
        Une liste de tuples (train_indices, val_indices) utilisable dans un GridSearchCV,
        où les indices sont les positions entières des lignes correspondant à `dates_series`.
    """
    if not isinstance(dates_series, pd.DatetimeIndex):
        raise ValueError("dates_series doit être un pd.DatetimeIndex.")

    years = dates_series.year # Accède directement à l'attribut .year, fonctionne pour DatetimeIndex et Series de datetimes
    unique_years = np.sort(years.unique())

    if len(unique_years) < min_train_years + val_years_duration:
        raise ValueError(
            f"Pas assez d'années dans le jeu de données ({len(unique_years)}) "
            f"pour un entraînement minimum de {min_train_years} ans "
            f"et une validation de {val_years_duration} an(s)."
        )

    all_possible_splits = []


    for i in range(min_train_years, len(unique_years) - val_years_duration + 1):
        train_years = unique_years[:i]
        val_years = unique_years[i : i + val_years_duration]

        # Récupère les positions entières des indices
        train_idx = np.where(np.isin(years, train_years))[0].tolist()
        val_idx = np.where(np.isin(years, val_years))[0].tolist()

        if not train_idx or not val_idx:
            continue # Skip if a split is empty

        all_possible_splits.append((train_idx, val_idx))

    if n_splits > len(all_possible_splits):
        print(f"Warning: n_splits ({n_splits}) is greater than the number of possible splits ({len(all_possible_splits)}). Returning all possible splits.")
        return all_possible_splits
    elif n_splits <= 0:
        raise ValueError("n_splits must be a positive integer.")
    else:
        return all_possible_splits[-n_splits:]

def train_val_split(folds_in: pd.DataFrame, val_duration = 90) -> tuple :
    """
        Tuple[pd.DataFrame]: A tuple of two dataframes (fold_train, fold_val)

        All fold_train should end on the month of may
        all fold val should end 3 months later

    """
    return



# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":
    print("--- Test de get_folds ---")

    # Création d'un DataFrame de test avec une colonne de dates
    test_dates_series = pd.date_range(start='2010-01-01', end='2020-12-31', freq='MS')
    # test_df = pd.DataFrame({'value': range(len(dates_index)), 'date_col': dates_index}) # No longer needed
    print(f"Dates disponibles dans la série: {test_dates_series.min().year}-{test_dates_series.max().year}")
    print(f"Shape de la série de dates: {test_dates_series.shape}")

    # Test 1: n_splits=3, min_train_years=3, val_years_duration=1
    print("\nTest 1: n_splits=3, min_train_years=3, val_years_duration=1")
    try:
        splits = get_folds(test_dates_series, n_splits=3, min_train_years=3, val_years_duration=1)
        for i, (train_idx, val_idx) in enumerate(splits):
            train_years = test_dates_series[train_idx].year.unique()
            val_years = test_dates_series[val_idx].year.unique()
            print(f"  Split {i+1}:")
            print(f"    Train years: {min(train_years)}-{max(train_years)}")
            print(f"    Val years:   {min(val_years)}-{max(val_years)}")
            print(f"    Train indices count: {len(train_idx)}, Val indices count: {len(val_idx)}")
            # Vérifier l'absence de chevauchement et l'ordre chronologique
            assert max(train_years) < min(val_years), "Fuite temporelle détectée !"
            assert len(val_years) == 1, "La durée de validation n'est pas respectée."
            assert len(train_years) >= 3, "Le nombre minimum d'années d'entraînement n'est pas respecté."
            assert max(train_idx) < min(val_idx), "Indices non chronologiques ou chevauchement détecté !"
    except ValueError as e:
        print(f"  Erreur attendue ou inattendue: {e}")

    # Test 2: n_splits=2, min_train_years=5, val_years_duration=2
    print("\nTest 2: n_splits=2, min_train_years=5, val_years_duration=2")
    try:
        splits = get_folds(test_dates_series, n_splits=2, min_train_years=5, val_years_duration=2)
        for i, (train_idx, val_idx) in enumerate(splits):
            train_years = test_dates_series[train_idx].year.unique()
            val_years = test_dates_series[val_idx].year.unique()
            print(f"  Split {i+1}:")
            print(f"    Train years: {min(train_years)}-{max(train_years)}")
            print(f"    Val years:   {min(val_years)}-{max(val_years)}")
            print(f"    Train indices count: {len(train_idx)}, Val indices count: {len(val_idx)}")
            assert max(train_years) < min(val_years), "Fuite temporelle détectée !"
            assert len(val_years) == 2, "La durée de validation n'est pas respectée."
            assert len(train_years) >= 5, "Le nombre minimum d'années d'entraînement n'est pas respecté."
            assert max(train_idx) < min(val_idx), "Indices non chronologiques ou chevauchement détecté !"
    except ValueError as e:
        print(f"  Erreur attendue ou inattendue: {e}")

    # Test 3: Pas assez d'années pour le split
    print("\nTest 3: Pas assez d'années (min_train_years trop grand)")
    try:
        get_folds(test_dates_series, n_splits=1, min_train_years=15, val_years_duration=1)
    except ValueError as e:
        print(f"  Erreur attendue: {e}")

    print("\n--- Fin des tests de get_folds ---")



## folding plot
if False:
    fig, ax = plt.subplots(figsize=(15, 8))

    # Définir les couleurs
    train_color = sns.color_palette("Blues")[2]
    val_color = sns.color_palette("Oranges")[2]

    # Pour chaque split, dessiner les barres d'entraînement et de validation
    for i, (train_idx, val_idx) in enumerate(splits):

    # Récupérer les dates min et max pour le train et la validation
    train_start_date = y_mensuel.index[train_idx].min()
    train_end_date = y_mensuel.index[train_idx].max()
    val_start_date = y_mensuel.index[val_idx].min()
    val_end_date = y_mensuel.index[val_idx].max()

    # Dessiner la barre d'entraînement
    ax.add_patch(mpatches.Rectangle(
        (train_start_date, i - 0.4), # (x, y) du coin inférieur gauche
        (train_end_date - train_start_date).days, # largeur en jours
        0.8, # hauteur
        facecolor=train_color,
        edgecolor='black',
        linewidth=0.5
    ))

    # Dessiner la barre de validation
    ax.add_patch(mpatches.Rectangle(
        (val_start_date, i - 0.4), # (x, y) du coin inférieur gauche
        (val_end_date - val_start_date).days, # largeur en jours
        0.8, # hauteur
        facecolor=val_color,
        edgecolor='black',
        linewidth=0.5
    ))

    # Ajouter des étiquettes pour les années de début/fin
    ax.text(train_start_date, i, f"{train_start_date.year}", va='center', ha='left', fontsize=8, color='black')
    ax.text(train_end_date, i, f"{train_end_date.year}", va='center', ha='right', fontsize=8, color='black')
    ax.text(val_start_date, i, f"{val_start_date.year}", va='center', ha='left', fontsize=8, color='black')
    ax.text(val_end_date, i, f"{val_end_date.year}", va='center', ha='right', fontsize=8, color='black')


    # Configuration de l'axe des x (temps)
    ax.set_xlim(dates_series_demo.min(), dates_series_demo.max())
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Pour avoir des années entières
    ax.tick_params(axis='x', rotation=45)

    # Configuration de l'axe des y (splits)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f'Split {i+1}' for i in range(len(splits))])
    ax.set_ylim(-0.5, len(splits) - 0.5) # Ajuster les limites pour que les barres soient bien centrées

    # Labels et titre
    ax.set_xlabel("Date")
    ax.set_ylabel("Split de Cross-Validation")
    ax.set_title(f"Visualisation des Splits de Cross-Validation Temporelle (Fenêtre Expansive Annuelle)\n"
                f"min_train_years={min_train_years}, val_years_duration={val_years_duration}, n_splits={n_splits_cv}")

    # Légende
    train_patch = mpatches.Patch(color=train_color, label='Entraînement')
    val_patch = mpatches.Patch(color=val_color, label='Validation')
    ax.legend(handles=[train_patch, val_patch], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

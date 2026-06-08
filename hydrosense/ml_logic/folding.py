import pandas as pd
import numpy as np
import random # Ajout pour la sélection aléatoire du mois de début de validation
from typing import List, Tuple

def get_folds(dates_series: pd.DatetimeIndex, n_splits: int = 5, min_train_years: int = 3, val_months_duration: int = 3):
    """
    Orchestre la génération d'indices de train/validation pour une cross-validation temporelle

    Exemple avec min_train_years=3,
    Si les dates vont de 2010 à 2020 et n_splits=3:
    Split 1: Train: 2010-2015, val: 2016 (assuming enough data for 3 splits)
    Split 2: Train: 2010-2016, val: 2017
    Split 3: Train: 2010-2017, val: 2018

    Arguments:
        dates_series (pd.DatetimeIndex): L'index temporel des données (X_train).
        n_splits (int): Le nombre de splits à générer (similaire à TimeSeriesSplit).
        min_train_years (int): Nombre minimum d'années pour le premier jeu d'entraînement.
                               Le premier split aura au moins `min_train_years` d'historique dans son train set.
        val_months_duration (int): Nombre de mois à utiliser pour chaque jeu de validation.
                                   Le mois de début de la validation sera aléatoirement Mars, Avril ou Mai.

    Retourne:
        Une liste de tuples (train_idx, val_idx) utilisable dans un GridSearchCV,
    """

    if not isinstance(dates_series, pd.DatetimeIndex):
        raise ValueError("dates_series doit être un pd.DatetimeIndex.")
    dates_series = dates_series.sort_values()

    # Obtenir les partitions annuelles (listes d'années de train, année de val)
    yearly_splits = split_years(dates_series, n_splits, min_train_years)
    # Et les affiner en fonctios du nb de mois
    # TODO gerer le cas où la donnée est hebdomadaire et non mensuelle
    final_splits = refine_split(dates_series, yearly_splits, val_months_duration, random_state=42)

    return final_splits


def split_years(dates_series: pd.DatetimeIndex, n_splits: int, min_train_years: int) -> List[Tuple[List[int], int]]:
    """
    Une sous-fonction de get_folds pour trouver les années de train et les années de validation pour chaque fold
    """
    years = dates_series.year
    unique_years = np.sort(years.unique())

    if len(unique_years) < min_train_years + 1:
        raise ValueError(
            f"Pas assez d'années dans le jeu de données ({len(unique_years)}) "
            f"pour un entraînement minimum de {min_train_years} ans "
            f"et au moins une année de validation."
        )

    all_possible_splits = []
    first_val_year_idx = min_train_years

    for i in range(first_val_year_idx, len(unique_years)):
        val_year = unique_years[i]
        train_years = unique_years[:i].tolist()

        if len(train_years) < min_train_years:
            continue

        all_possible_splits.append((train_years, val_year))

    if n_splits > len(all_possible_splits):
        print(f"Warning: n_splits ({n_splits}) is greater than the number of possible splits ({len(all_possible_splits)}). Returning all possible splits.")
        return all_possible_splits
    elif n_splits <= 0:
        raise ValueError("n_splits must be a positive integer.")
    else:
        return all_possible_splits[-n_splits:]


def refine_split(
    dates_series: pd.DatetimeIndex,
    yearly_splits: List[Tuple[List[int], int]],
    val_months_duration: int,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Sous fonction de get_folds qui termine le travail reprenant les folds de split_years
    selectionne les val_start_month, tirage au sort [3,4,5] , random_state = 42
    ralonge la periode d'entrainement pour qu'elle se termine juste avant la validation
    Fournit les liste d indices (train_idx, val_idx) en tuple
    """
    random.seed(random_state)
    final_splits = []
    possible_val_start_months = [3, 4, 5, 6] # Mars, Avril, Mai, juin

    for train_years, val_year in yearly_splits:
        val_start_month = random.choice(possible_val_start_months)
        val_start_date = pd.Timestamp(year=val_year, month=val_start_month, day=1)

        val_end_date = val_start_date + pd.DateOffset(months=val_months_duration) - pd.DateOffset(days=1)
        train_end_date = val_start_date - pd.DateOffset(days=1)

        if val_end_date > dates_series.max():
            print(f"Warning: Validation period for year {val_year} (ending {val_end_date.date()}) "
                  f"exceeds data end date ({dates_series.max().date()}). Skipping this split.")
            continue

        train_idx = np.where(dates_series <= train_end_date)[0].tolist()
        val_idx = np.where((dates_series >= val_start_date) & (dates_series <= val_end_date))[0].tolist()

        if not train_idx or not val_idx:
            continue

        final_splits.append((train_idx, val_idx))

    return final_splits



# ================================================================
# TESTS
# ================================================================

if __name__ == "__main__":
    print("--- Test de get_folds (avec mois de début aléatoire) ---")

    # Création d'un DataFrame de test avec une colonne de dates
    test_dates_series = pd.date_range(start='2010-01-31', end='2020-12-31', freq='ME') # Utilise la fréquence Month End pour la cohérence
    # test_df = pd.DataFrame({'value': range(len(dates_index)), 'date_col': dates_index}) # No longer needed
    print(f"Dates disponibles dans la série: {test_dates_series.min().year}-{test_dates_series.max().year}")
    print(f"Shape de la série de dates: {test_dates_series.shape}")

    # Test 1: n_splits=3, min_train_years=3, val_months_duration=3
    print("\nTest 1: n_splits=3, min_train_years=3, val_months_duration=3")
    try:
        splits = get_folds(test_dates_series, n_splits=3, min_train_years=3, val_months_duration=3)
        for i, (train_idx, val_idx) in enumerate(splits):
            train_period = test_dates_series[train_idx]
            val_period = test_dates_series[val_idx]
            print(f"  Split {i+1}:")
            print(f"    Train: {train_period.min().strftime('%Y-%m')} à {train_period.max().strftime('%Y-%m')} ({len(train_idx)} samples)")
            print(f"    Val:   {val_period.min().strftime('%Y-%m')} à {val_period.max().strftime('%Y-%m')} ({len(val_idx)} samples)")
            print(f"    Train indices count: {len(train_idx)}, Val indices count: {len(val_idx)}")
            # Assertions pour vérifier l'absence de chevauchement et l'ordre chronologique
            assert train_period.max() < val_period.min(), "Fuite temporelle détectée !"
            assert len(val_period) == 3, "La durée de validation n'est pas respectée (3 mois)."
            assert (train_period.max().year - train_period.min().year + 1) >= 3, "Le nombre minimum d'années d'entraînement n'est pas respecté."
            assert max(train_idx) < min(val_idx), "Indices non chronologiques ou chevauchement détecté !"
    except ValueError as e:
        print(f"  Erreur attendue ou inattendue: {e}")

    # Test 2: n_splits=2, min_train_years=5, val_years_duration=2
    print("\nTest 2: n_splits=25, min_train_years=5, val_months_duration=4")
    try:
        splits = get_folds(test_dates_series, n_splits=25, min_train_years=5, val_months_duration=4)
        for i, (train_idx, val_idx) in enumerate(splits):
            train_period = test_dates_series[train_idx]
            val_period = test_dates_series[val_idx]
            print(f"  Split {i+1}:")
            print(f"    Train: {train_period.min().strftime('%Y-%m')} à {train_period.max().strftime('%Y-%m')} ({len(train_idx)} samples)")
            print(f"    Val:   {val_period.min().strftime('%Y-%m')} à {val_period.max().strftime('%Y-%m')} ({len(val_idx)} samples)")
            print(f"    Train indices count: {len(train_idx)}, Val indices count: {len(val_idx)}")
            assert train_period.max() < val_period.min(), "Fuite temporelle détectée !" # Vérifie l'ordre chronologique
            assert len(val_period) == 4, "La durée de validation n'est pas respectée (4 mois)." # Vérifie le nombre exact de mois
            # La vérification du nombre d'années min est gérée dans split_years, mais on peut la revérifier ici
            assert max(train_idx) < min(val_idx), "Indices non chronologiques ou chevauchement détecté !"
    except ValueError as e:
        print(f"  Erreur attendue ou inattendue: {e}")

    # Test 3: Pas assez d'années pour le split
    print("\nTest 3: Pas assez d'années (min_train_years trop grand)")
    try:
        get_folds(test_dates_series, n_splits=1, min_train_years=15, val_months_duration=3)
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

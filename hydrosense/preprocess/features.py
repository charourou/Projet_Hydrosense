import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.utils.geo import trouver_voisins_hydrogeologiques




def extraire_PCA_piezo( voisins_ids: list,
                                    n_components: int = 1,
                                    fenetre_lissage: int = 7,
                                    toggle_plot: bool = False
                                    ) -> pd.DataFrame:
    """
    Extrait les composantes principales (PCA) des piézomètres voisins.

    Étapes :
    1. Récupération et alignement temporel.
    2. Imputation des valeurs manquantes (IterativeImputer).
    3. Lissage temporel (Moyenne roulante).
    4. Standardisation et extraction ACP.
    """
    print("🔄 Récupération et alignement des voisins...")
    series_list = []

    # ALIGNEMENT TEMPOREL
    for bss_id in voisins_ids:
        df_voisin = clean_piezo(load_piezo_bq(bss_id=bss_id))
        df_voisin['date_mesure'] = pd.to_datetime(df_voisin['date_mesure']).dt.normalize()

        # On indexe par la date et on isole la chronique
        s_voisin = df_voisin.set_index('date_mesure')['niveau_nappe_eau'].rename(bss_id)
        series_list.append(s_voisin)

    df_wide = pd.concat(series_list, axis=1)

    print(f"📊 Dimensions de la matrice fusionnée : {df_wide.shape}")
    print(f"⚠️ Pourcentage de valeurs manquantes (brut) : {(df_wide.isna().sum().sum() / df_wide.size * 100):.1f}%")

    print("🧠 Imputation intelligente des trous...")
    imputer = IterativeImputer(max_iter=10, random_state=42)
    matrice_imputee = imputer.fit_transform(df_wide)
    df_imputed = pd.DataFrame(matrice_imputee, index=df_wide.index, columns=df_wide.columns)


    if fenetre_lissage > 1:
        print(f"🌊 Application du lissage (Moyenne roulante sur {fenetre_lissage} jours)...")
        # min_periods=1 empêche de créer des NaNs au tout début de la série
        df_imputed = df_imputed.rolling(window=fenetre_lissage, min_periods=1).mean()

    print("📉 Calcul de l'ACP...")
    scaler = StandardScaler()
    matrice_scaled = scaler.fit_transform(df_imputed)

    pca = PCA(n_components=n_components)
    matrice_pca = pca.fit_transform(matrice_scaled)

    # Création du DataFrame de sortie dynamique selon le nombre de composantes
    colonnes_pca = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(matrice_pca, index=df_wide.index, columns=colonnes_pca)

    variance_totale = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance expliquée par les {n_components} axes : {variance_totale:.1f}%")

    if toggle_plot:
        plt.figure(figsize=(12, 5))
        for col in colonnes_pca:
            plt.plot(df_pca.index, df_pca[col], label=col, alpha=0.7)
        plt.title(f"Les {n_components} composantes principales régionales (Lissées sur {fenetre_lissage}j)")
        plt.xlabel("Date")
        plt.ylabel("Valeur de la composante (Standardisée)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df_pca

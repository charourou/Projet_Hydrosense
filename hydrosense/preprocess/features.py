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

    n_features = matrice_scaled.shape[1]  # Nombre réel de voisins valides
    n_comp_calculees = min(n_components, n_features)

    if n_comp_calculees > 0:
        pca = PCA(n_components=n_comp_calculees)
        matrice_pca = pca.fit_transform(matrice_scaled)

        colonnes_pca_calculees = [f'PC{i+1}' for i in range(n_comp_calculees)]
        df_pca = pd.DataFrame(matrice_pca, index=df_wide.index, columns=colonnes_pca_calculees)

        variance_totale = pca.explained_variance_ratio_.sum() * 100
        print(f"✅ Variance expliquée par les {n_comp_calculees} axes : {variance_totale:.1f}%")
    else:
        # Cas extrême : 0 voisin valide
        df_pca = pd.DataFrame(index=df_wide.index)

    # On s'assure que le DataFrame 3 colonnes PC1, PC2, PC3
    for i in range(n_components):
        nom_col = f'PC{i+1}_hydro'
        if nom_col not in df_pca.columns:
            df_pca[nom_col] = 0.0  # On remplit avec une valeur neutre
            print(f"⚠️ Création d'une colonne {nom_col} vide (pas assez de voisins).")

    if toggle_plot:
        plt.figure(figsize=(12, 5))
        for col in colonnes_pca_calculees:
            plt.plot(df_pca.index, df_pca[col], label=col, alpha=0.7)
        plt.title(f"Les {n_components} composantes principales régionales (Lissées sur {fenetre_lissage}j)")
        plt.xlabel("Date")
        plt.ylabel("Valeur de la composante (Standardisée)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return df_pca

def enrichir_contexte_regional(df_cible: pd.DataFrame,
                               bss_cible: str,
                               df_catalogue: pd.DataFrame,
                               n_voisins: int = 10) -> pd.DataFrame:
    """
    Orchestre les Étapes 4 et 5 :
    - Trouve les voisins hydrogéologiques.
    - Calcule l'ACP de leurs niveaux d'eau bruts.
    - Fusionne les composantes (PC1, PC2, PC3) avec le DataFrame cible.
    """
    print(f"\n🌍 ÉTAPE 4 : Recherche du contexte régional pour {bss_cible}...")

    try:
        top_voisins = trouver_voisins_hydrogeologiques(
                                        df_catalogue=df_catalogue,
                                        bss_target=bss_cible,
                                        n_voisins=n_voisins
                                                        )
        voisins_ids = top_voisins['bss_id'].values

        if len(voisins_ids) == 0:
            print("⚠️ Aucun voisin pertinent trouvé.")
            return df_cible

        print(f"   ✔️ {len(voisins_ids)} voisins trouvés. Calcul de l'ACP...")

        # Extraction des features
        df_pca = extraire_PCA_piezo(
                                    voisins_ids=voisins_ids,
                                    n_components=3,
                                    )

        print("\n🔗 ÉTAPE 5 : Fusion finale des données (Merge)...")

        # df_pca a la date en index, vraiment ??
        # est-ce que nous allons avoir un problème
        df_enrichi = df_cible.merge(
                                    df_pca,
                                    left_on='date_mesure',
                                    right_index=True,
                                    how='left'
                                    )

        print(f"✅ Contexte régional ajouté avec succès ({len(df_pca.columns)} nouvelles colonnes).")
        return df_enrichi

    except Exception as e:
        print(f"❌ Erreur critique lors de l'intégration du contexte régional : {e}")
        return df_cible

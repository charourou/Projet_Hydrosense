import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIGMA = 4.903e-9 # Constante de Stefan-Boltzmann


def estim_PU(P_mm, T_moy_C, vent_m_s, lat_deg, altitude_m, jour_annee, humidite_relative=60.0):
    """Version vectorisée de Penman-Monteith"""

    altitude_m = np.clip(altitude_m, 0, 1000) # précaution

    P_atm = 101.3 * ((293 - 0.0065 * altitude_m) / 293) ** 5.26
    gamma = 0.000665 * P_atm

    e_s = 0.6108 * np.exp(17.27 * T_moy_C / (T_moy_C + 237.3))
    e_a = e_s * (humidite_relative / 100.0)
    delta = (4098 * e_s) / ((T_moy_C + 237.3) ** 2)

    lat_rad = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * jour_annee / 365)
    declinaison_sol = 0.409 * np.sin(2 * np.pi * jour_annee / 365 - 1.39)


    x = np.clip(-np.tan(lat_rad) * np.tan(declinaison_sol), -1.0, 1.0)
    omega_s = np.arccos(x)

    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        omega_s * np.sin(lat_rad) * np.sin(declinaison_sol) +
        np.cos(lat_rad) * np.cos(declinaison_sol) * np.sin(omega_s)
    )

    Rs = 0.50 * Ra
    Rns = (1 - 0.23) * Rs
    Rnl = SIGMA * ((T_moy_C + 273.16)**4) * (0.34 - 0.14 * np.sqrt(e_a)) * (1.35 * (Rs / (0.75 * Ra)) - 0.35)
    Rn = Rns - Rnl

    terme_radiatif = 0.408 * delta * Rn
    terme_advectif = gamma * (900 / (T_moy_C + 273.16)) * vent_m_s * (e_s - e_a)
    denominateur = delta + gamma * (1 + 0.34 * vent_m_s)

    eto = (terme_radiatif + terme_advectif) / denominateur
    eto = np.maximum(0, eto) # Remplace le max() pour garder la valeur >= 0

    return np.round(P_mm - eto, 3)




def lag_pluie(df: pd.DataFrame, col_pluie: str = 'PU_synth', col_nappe: str = 'niveau_nappe_eau',
                           max_lag: int = 120,
                           toggle_plot: bool = False,
                           ax = None
                           ) -> int:
    """
    Calcule la corrélation croisée entre un signal météo et un niveau de nappe
    pour estimer le temps de réponse (inertie) de l'aquifère.

    Retourne :
        int : Le décalage (en jours) maximisant la corrélation.
    """

    # 1. Préparation des données (sécurité contre les NaN)
    df_corr = df[[col_pluie, col_nappe]].dropna().copy()

    if df_corr.empty:
        print("❌ Erreur : Le DataFrame est vide après suppression des NaN.")
        return 0

    lags = range(0, max_lag + 1)
    corr_values = []

    # Calcul des corrélations décalées
    for lag in lags:
        corr = df_corr[col_pluie].corr(df_corr[col_nappe].shift(-lag))
        corr_values.append(corr)

    # Identification du meilleur lag
    best_idx = np.argmax(corr_values)
    best_lag = lags[best_idx]
    best_corr = corr_values[best_idx]

    if toggle_plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            is_standalone = True

        else:
            is_standalone = False

        ax.plot(lags, corr_values, marker='o', color='teal', markersize=4,
                label='Corrélation de Pearson')
        ax.plot(best_lag, best_corr, marker='*', color='red', markersize=12,
                label=f'Max: {best_lag} jours')

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        ax.set_title(f"Temps de réponse de la nappe : {col_pluie} vs {col_nappe}")
        ax.set_xlabel("Décalage (Jours)", fontsize=12)
        ax.set_ylabel("Coefficient de Corrélation", fontsize=12)
        ax.legend(loc = 'upper right', frameon= True)
        ax.grid(True, alpha=0.5, linestyle=':' )

        # Nettoyage des bordures
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')

        if is_standalone:
            plt.tight_layout()
            plt.show()

    print(f"✅ Le temps de réponse optimal estimé de la nappe est de {best_lag} jours (Corrélation: {best_corr:.3f}).")

    return best_lag, best_corr, ax

# --- Exemple d'utilisation ---
# temps_reponse = calculer_temps_reponse(df, col_pluie='PU_synth', col_nappe='niveau_nappe_eau', max_lag=90)

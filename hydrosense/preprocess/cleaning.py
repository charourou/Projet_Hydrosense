import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_piezo(df: pd.DataFrame,
                col_date: str = "date_mesure"):

    # — Étape 1 : si trous > 50 jours,
    # on repart de la nouvelle date
    gaps = df[col_date].diff().dt.days
    max_gap_days = 50
    big_gaps = gaps[gaps > max_gap_days]

    if not big_gaps.empty:
        last_gap_idx = big_gaps.index[-1]
        last_gap_date = df.loc[last_gap_idx, col_date]
        n_dropped = last_gap_idx
        df = df.iloc[last_gap_idx:].copy()
        print(f"Trou de {int(big_gaps.iloc[-1])} jours détecté ! {n_dropped} lignes supprimées.")


    # Rééchantillonner à fréquence journalière — sans mettre col date dans l'index
    df = df.set_index(col_date)
    df = df["niveau_nappe_eau"].resample("D").mean().to_frame()

    # combler les trous restants —
    df = df.interpolate(method="time")
    df = df.reset_index()

    # remettre la date en colonne
    df.rename(columns={"index": col_date}, inplace=True)

    print(f"DataFrame final : {df[col_date].iloc[0]} → {df[col_date].iloc[-1]} | {len(df)} jours")

    return df

def clean_piezo2(df: pd.DataFrame, col_date: str = "date_mesure"):

    # — Étape 1 : si trous > 50 jours, on repart de la nouvelle date
    gaps = df[col_date].diff().dt.days
    max_gap_days = 50
    big_gaps = gaps[gaps > max_gap_days]

    if not big_gaps.empty:
        last_gap_idx = big_gaps.index[-1]
        n_dropped = last_gap_idx
        df = df.iloc[last_gap_idx:].copy()
        print(f"Trou de {int(big_gaps.iloc[-1])} jours détecté ! {n_dropped} lignes supprimées.")

    # — Étape 2 : Rééchantillonner à fréquence journalière avec l'index
    df = df.set_index(col_date)

    # 💡 CORRECTION 1 : On utilise "sum" (ou "mean") pour la pluie, "RR" faisait planter
    df_journalier = df.resample("D").agg({
        "niveau_nappe_eau": "mean",
        "RR_synth": "sum"  # Remplacé "RR" par "sum" pour sommer la pluie du jour
    })
    print (df_journalier)

    # — Étape 3 : Combler les trous restants
    # 💡 CORRECTION 2 : On applique l'interpolation sur "df_journalier" et on limite à 7 jours pour la nappe
    df_journalier["niveau_nappe_eau"] = df_journalier["niveau_nappe_eau"].interpolate(method="time", limit=7)

    # Pour la pluie, s'il y a un trou, on remplace par 0 (pas de pluie) plutôt que d'interpoler une ligne droite
    df_journalier["RR_synth"] = df_journalier["RR_synth"].fillna(0)

    # — Étape 4 : Remettre la date en colonne et nettoyer l'index
    df_final = df_journalier.reset_index()

    # Sécurité sur le nom de la colonne de date après le reset_index
    if "index" in df_final.columns:
        df_final.rename(columns={"index": col_date}, inplace=True)

    print(f"DataFrame final : {df_final[col_date].iloc[0].date()} → {df_final[col_date].iloc[-1].date()} | {len(df_final)} jours")

    return df_final

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_piezo(df: pd.DataFrame): 

    # — Étape 1 : si trous > 50 jours , on repart de la nouvelle date
    gaps = df.index.to_series().diff().dt.days
    max_gap_days = 50

    big_gaps = gaps[gaps > max_gap_days]

    if big_gaps.empty:
        print("Aucun trou > 50 jours détecté !")
    else:
        last_gap_date = big_gaps.index[-1]
        n_dropped = (df.index < last_gap_date).sum()
        df = df[df.index >= last_gap_date].copy()
        print(f"Trou de {int(big_gaps.iloc[-1])} jours détecté !")
        print(f"{n_dropped} lignes antérieures à {last_gap_date.date()} supprimées !")

    # — Étape 2 : rééchantillonner à fréquence journalière —
    df = df["niveau_nappe_eau"].resample("D").mean().to_frame()

    # — Étape 3 : combler les trous restants —
    df = df.interpolate(method="time")

    print(f"DataFrame final : {df.index[0].date()} → {df.index[-1].date()} | {len(df)} jours")
    
    return df
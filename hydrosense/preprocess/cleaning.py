import pandas as pd

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

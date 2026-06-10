import streamlit as st
import pandas as pd
from utils.api_client import load_catalog_interm

st.title("🗺️ Catalogue des piézomètres")

df = load_catalog_interm()

# ── KPIs ──────────────────────────────────────────────────────────────────────
# col1, col2, col3 = st.columns(3)
# col1.metric("Piézomètres", f"{len(df):,}".replace(",", " "))
# col2.metric("Départements", df["code_departement"].nunique())
# col3.metric("Communes", df["nom_commune"].nunique())

# st.divider()

# ── Filtres ───────────────────────────────────────────────────────────────────
f1, f2 = st.columns([1, 2])
with f1:
    depts = sorted(df["code_departement"].dropna().unique())
    dept_sel = st.selectbox("Département", ["Tous"] + list(depts))
with f2:
    search = st.text_input("Recherche", placeholder="BSS ID ou commune…")

df_view = df.copy()
if dept_sel != "Tous":
    df_view = df_view[df_view["code_departement"] == dept_sel]
if search:
    mask = (
        df_view["bss_id"].str.contains(search, case=False, na=False)
        | df_view["nom_commune"].str.contains(search, case=False, na=False)
    )
    df_view = df_view[mask]

# ── Colonnes à afficher ───────────────────────────────────────────────────────
COLS = {
    "bss_id":            "BSS ID",
    "nom_commune":       "Commune",
    "code_departement":  "Dép.",
    "nom_departement":   "Département",
    "altitude_station":  "Altitude (m)",
}
available = [c for c in COLS if c in df_view.columns]
df_display = df_view[available].rename(columns=COLS).reset_index(drop=True)

st.caption(f"{len(df_display)} piézomètre(s) affiché(s)")

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "BSS ID":        st.column_config.TextColumn(width="medium"),
        "Commune":       st.column_config.TextColumn(width="medium"),
        "Dép.":          st.column_config.TextColumn(width="small"),
        "Département":   st.column_config.TextColumn(width="medium"),
        "Altitude (m)":  st.column_config.NumberColumn(format="%.0f m", width="small"),
    },
)

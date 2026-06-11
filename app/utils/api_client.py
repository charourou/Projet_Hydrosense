"""
utils/api_client.py
───────────────────
Fonctions d'accès aux données pour l'app Hydro-Sense.
Toutes les données transitent par l'API FastAPI (API_URL).
Les signatures sont identiques à l'ancienne version BQ directe.
"""

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

API_URL = os.getenv("API_URL", "https://hydrosense-api-121714908762.europe-west1.run.app")

# Noms de colonnes percentile — source unique de vérité (miroir de fast.py)
PERCENTILE_COLS = {
    "p95": "p95_global",
    "p85": "p85_global",
    "p20": "p20_global",
    "p10": "p10_global",
    "p5":  "p5_global",
}


def _get(path: str, **params) -> requests.Response:
    response = requests.get(f"{API_URL}{path}", params=params or None)
    response.raise_for_status()
    return response


# ── Catalogue ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.DataFrame(_get("/catalogue/raw").json())


@st.cache_data
def load_catalog_map() -> pd.DataFrame:
    return pd.DataFrame(_get("/catalogue/map").json())


@st.cache_data
def load_catalog_interm() -> pd.DataFrame:
    return pd.DataFrame(_get("/catalogue").json())


@st.cache_data
def load_catalog_ml() -> pd.DataFrame:
    """Catalogue filtré sur les piézomètres avec données ML (chroniques_plean)."""
    return pd.DataFrame(_get("/catalogue/ml").json())


@st.cache_data
def load_catalog_ml_map() -> pd.DataFrame:
    """Piézomètres ML (France entière) avec coordonnées GPS et seuils percentiles."""
    return pd.DataFrame(_get("/catalogue/ml/map").json())


@st.cache_data
def load_single_piezo_map(bss_id: str) -> pd.DataFrame:
    try:
        row = _get(f"/piezo/{bss_id}/map").json()
        return pd.DataFrame([row])
    except requests.HTTPError:
        return pd.DataFrame()


# ── Seuils ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_seuils_interm(bss_id: str) -> dict | None:
    try:
        return _get(f"/piezo/{bss_id}/seuils").json()
    except requests.HTTPError:
        return None


@st.cache_data
def load_catalog_map_dept(code_departement: str = "79") -> pd.DataFrame:
    df = pd.DataFrame(_get(f"/catalogue/map/dept/{code_departement}").json())
    # Renommer les colonnes percentile vers les noms courts attendus par seuils_from_row
    rename = {v: v for v in PERCENTILE_COLS.values()}  # déjà au bon format
    return df


def seuils_from_row(row: pd.Series) -> dict | None:
    """Extrait un dict {p5, p10, p20, p85, p95} depuis une ligne de load_catalog_map_dept()."""
    cols = list(PERCENTILE_COLS.values())
    if row[cols].isnull().any():
        return None
    return {key: float(row[col]) for key, col in PERCENTILE_COLS.items()}


# ── Précipitations ────────────────────────────────────────────────────────────

@st.cache_data
def load_pluie(bss_id: str, days: int = 30) -> pd.DataFrame:
    """Précipitations (pluie_mm) depuis chroniques_plean via l'API."""
    try:
        data = _get(f"/piezo/{bss_id}/pluie", days=days).json()
        df = pd.DataFrame(data["pluie"])
        df.rename(columns={"date": "date_mesure"}, inplace=True)
        df["date_mesure"] = pd.to_datetime(df["date_mesure"])
        return df
    except requests.HTTPError:
        return pd.DataFrame(columns=["date_mesure", "pu_synth"])


# ── Historique ────────────────────────────────────────────────────────────────

@st.cache_data
def load_historique(bss_id: str) -> pd.DataFrame:
    """Historique piézométrique nettoyé via l'API."""
    data = _get(f"/piezo/{bss_id}/historique").json()
    df = pd.DataFrame(data["historique"])
    df.rename(columns={"date": "date_mesure", "niveau": "niveau_nappe_eau"}, inplace=True)
    df["date_mesure"] = pd.to_datetime(df["date_mesure"])
    return df

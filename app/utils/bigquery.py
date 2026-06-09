"""
utils/bigquery.py
─────────────────
Fonctions d'accès BigQuery pour l'app Hydro-Sense.

Bonnes pratiques appliquées :
- Client BQ instancié une seule fois au niveau module (pas dans chaque fn)
- Colonnes percentile centralisées dans PERCENTILE_COLS
- Toutes les fonctions retournent un type annoté
- @st.cache_data sur toutes les fonctions de lecture
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(override=True)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET        = os.getenv("BQ_DATASET_ID")

# ── Client unique partagé ─────────────────────────────────────────────────────
_client = bigquery.Client(project=GCP_PROJECT_ID)

# ── Noms de colonnes percentile — source unique de vérité ────────────────────
PERCENTILE_COLS = {
    "p95": "p95_global",
    "p85": "p85_global",
    "p20": "p20_global",
    "p10": "p10_global",
    "p5":  "p5_global",
}


def _table(name: str) -> str:
    """Retourne le chemin complet d'une table BQ."""
    return f"`{GCP_PROJECT_ID}.{DATASET}.{name}`"


# ── Catalogue ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_catalog() -> pd.DataFrame:
    query = f"SELECT * FROM {_table('cat_piezo_raw')}"
    return _client.query(query).to_dataframe()


@st.cache_data
def load_catalog_map() -> pd.DataFrame:
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM {_table('cat_piezo_raw')}
        WHERE x IS NOT NULL AND y IS NOT NULL
    """
    return _client.query(query).to_dataframe()


@st.cache_data
def load_catalog_interm() -> pd.DataFrame:
    """
    Charge la liste des piézomètres depuis cat_piezo_interm
    pour alimenter le sélecteur de l'interface.
    """
    query = f"""
        SELECT i.bss_id, i.nom_commune, i.code_departement, r.nom_departement
        FROM {_table('cat_piezo_interm')} i
        LEFT JOIN {_table('cat_piezo_raw')} r USING (bss_id)
        WHERE i.bss_id IS NOT NULL
        ORDER BY i.code_departement, i.nom_commune
    """
    return _client.query(query).to_dataframe()


@st.cache_data
def load_single_piezo_map(bss_id: str) -> pd.DataFrame:
    """Coordonnées et infos d'un seul piézomètre."""
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM {_table('cat_piezo_raw')}
        WHERE bss_id = @bss_id
          AND x IS NOT NULL AND y IS NOT NULL
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)]
    )
    return _client.query(query, job_config=job_config).to_dataframe()


# ── Seuils ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_seuils_interm(bss_id: str) -> dict | None:
    """
    Charge les seuils percentiles depuis cat_piezo_interm.

    Retourne None si le piézomètre est introuvable ou si
    l'une des valeurs percentile est nulle (déclenche le fallback côté UI).
    """
    cols_sql = ", ".join(PERCENTILE_COLS.values())
    query = f"""
        SELECT {cols_sql}
        FROM {_table('cat_piezo_interm')}
        WHERE bss_id = @bss_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)]
    )
    df = _client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        return None

    row = df.iloc[0]
    if row[list(PERCENTILE_COLS.values())].isnull().any():
        return None

    # Remapper les noms de colonnes BQ vers les clés courtes (p95, p85, …)
    return {key: float(row[col]) for key, col in PERCENTILE_COLS.items()}


@st.cache_data
def load_seuils(bss_id: str) -> dict | None:
    """
    Charge les 4 seuils réglementaires depuis chroniques_piezo.

    Retourne None si le bss_id n'est pas trouvé.
    """
    query = f"""
        SELECT seuil_vigilance, seuil_alerte, seuil_alerte_renforcee, seuil_crise
        FROM {_table('chroniques_piezo')}
        WHERE bss_id = @bss_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)]
    )
    df = _client.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        return None

    row = df.iloc[0]
    return {
        "vigilance":        float(row["seuil_vigilance"]),
        "alerte":           float(row["seuil_alerte"]),
        "alerte_renforcee": float(row["seuil_alerte_renforcee"]),
        "crise":            float(row["seuil_crise"]),
    }


@st.cache_data
def load_catalog_map_dept(code_departement: str = "79") -> pd.DataFrame:
    """
    Charge les piézomètres d'un département depuis cat_piezo_interm + cat_piezo_raw
    en UNE SEULE requête BQ (coordonnées + seuils percentiles).

    Retourne un DataFrame avec colonnes :
        bss_id, nom_commune, code_departement, x, y,
        p95_global, p85_global, p20_global, p10_global, p5_global
    """
    cols_sql = ", ".join(f"i.{c}" for c in PERCENTILE_COLS.values())
    query = f"""
        SELECT
            i.bss_id,
            i.nom_commune,
            i.code_departement,
            r.x,
            r.y,
            {cols_sql}
        FROM {_table('cat_piezo_interm')} i
        JOIN {_table('cat_piezo_raw')} r USING (bss_id)
        WHERE i.code_departement = @dept
          AND r.x IS NOT NULL
          AND r.y IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dept", "STRING", code_departement)
        ]
    )
    return _client.query(query, job_config=job_config).to_dataframe()


def seuils_from_row(row: pd.Series) -> dict | None:
    """
    Extrait un dict {p5, p10, p20, p85, p95} depuis une ligne
    du DataFrame retourné par load_catalog_map_dept().
    Retourne None si l'une des valeurs percentile est nulle.
    """
    cols = list(PERCENTILE_COLS.values())
    if row[cols].isnull().any():
        return None
    return {key: float(row[col]) for key, col in PERCENTILE_COLS.items()}
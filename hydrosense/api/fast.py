import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import bigquery

from hydrosense.database.bigquery import load_piezo_bq, load_plean
from hydrosense.params import BQ_DATASET_ID, GCP_PROJECT_ID
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.preprocess.preprocessor import preprocess_week
from hydrosense.interface.main import train, pred_future

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bq = bigquery.Client(project=GCP_PROJECT_ID)

PERCENTILE_COLS = {
    "p95": "p95_global",
    "p85": "p85_global",
    "p20": "p20_global",
    "p10": "p10_global",
    "p5":  "p5_global",
}


# ══════════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"greeting": "Hello"}


# ══════════════════════════════════════════════════════════════════════════════
# CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/catalogue")
def catalogue():
    """Liste des piézomètres pour le sélecteur UI (bss_id, commune, dept)."""
    query = f"""
        SELECT i.bss_id, i.nom_commune, i.code_departement, r.nom_departement
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_interm` i
        LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw` r USING (bss_id)
        WHERE i.bss_id IS NOT NULL
        ORDER BY i.code_departement, i.nom_commune
    """
    df = _bq.query(query).to_dataframe()
    return json.loads(df.to_json(orient="records"))


@app.get("/catalogue/ml/map")
def catalogue_ml_map():
    """Piézomètres ML (France entière) avec coordonnées GPS et seuils percentiles."""
    cols_sql = ", ".join(f"i.{c}" for c in PERCENTILE_COLS.values())
    query = f"""
        SELECT
            i.bss_id, i.nom_commune, i.code_departement,
            r.x, r.y,
            {cols_sql}
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_interm` i
        JOIN `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw` r USING (bss_id)
        WHERE i.bss_id IN (
            SELECT DISTINCT bss_id FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.chroniques_plean`
        )
          AND r.x IS NOT NULL AND r.y IS NOT NULL
    """
    df = _bq.query(query).to_dataframe()
    return json.loads(df.to_json(orient="records"))


@app.get("/catalogue/ml")
def catalogue_ml():
    """Piézomètres ayant des données dans chroniques_plean (prévision disponible)."""
    query = f"""
        SELECT i.bss_id, i.nom_commune, i.code_departement, r.nom_departement
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_interm` i
        LEFT JOIN `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw` r USING (bss_id)
        WHERE i.bss_id IN (
            SELECT DISTINCT bss_id FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.chroniques_plean`
        )
          AND i.bss_id IS NOT NULL
        ORDER BY i.code_departement, i.nom_commune
    """
    df = _bq.query(query).to_dataframe()
    return json.loads(df.to_json(orient="records"))


@app.get("/catalogue/raw")
def catalogue_raw():
    """Catalogue complet depuis cat_piezo_raw."""
    query = f"SELECT * FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw`"
    df = _bq.query(query).to_dataframe()
    return json.loads(df.to_json(orient="records"))


@app.get("/catalogue/map")
def catalogue_map():
    """Tous les piézomètres avec coordonnées GPS."""
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw`
        WHERE x IS NOT NULL AND y IS NOT NULL
    """
    df = _bq.query(query).to_dataframe()
    return json.loads(df.to_json(orient="records"))


@app.get("/catalogue/map/dept/{code_departement}")
def catalogue_map_dept(code_departement: str):
    """Piézomètres d'un département avec coordonnées GPS et seuils percentiles."""
    cols_sql = ", ".join(f"i.{c}" for c in PERCENTILE_COLS.values())
    query = f"""
        SELECT
            i.bss_id, i.nom_commune, i.code_departement,
            r.x, r.y,
            {cols_sql}
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_interm` i
        JOIN `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw` r USING (bss_id)
        WHERE i.code_departement = @dept
          AND r.x IS NOT NULL AND r.y IS NOT NULL
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("dept", "STRING", code_departement)]
    )
    df = _bq.query(query, job_config=job_config).to_dataframe()
    return json.loads(df.to_json(orient="records"))


# ══════════════════════════════════════════════════════════════════════════════
# PIÉZOMÈTRE INDIVIDUEL
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/piezo/{bss_id}/historique")
def historique(bss_id: str):
    """Historique piézométrique nettoyé (date + niveau)."""
    try:
        df_raw = load_piezo_bq(bss_id)
        df_clean = clean_piezo(df_raw)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "bss_id": bss_id,
        "historique": [
            {"date": row.date_mesure.strftime("%Y-%m-%d"), "niveau": round(float(row.niveau_nappe_eau), 3)}
            for row in df_clean.itertuples()
        ],
    }


@app.get("/piezo/{bss_id}/map")
def piezo_map(bss_id: str):
    """Coordonnées et infos géographiques d'un piézomètre."""
    query = f"""
        SELECT bss_id, nom_commune, nom_departement, x, y
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_raw`
        WHERE bss_id = @bss_id
          AND x IS NOT NULL AND y IS NOT NULL
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)]
    )
    df = _bq.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Piézomètre {bss_id} introuvable ou sans coordonnées")
    return json.loads(df.to_json(orient="records"))[0]


# TODO: seuils à calculer mois par mois (et non une valeur annuelle)
@app.get("/piezo/{bss_id}/seuils")
def seuils(bss_id: str):
    """Seuils percentiles (p5, p10, p20, p85, p95) d'un piézomètre."""
    cols_sql = ", ".join(PERCENTILE_COLS.values())
    query = f"""
        SELECT {cols_sql}
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.cat_piezo_interm`
        WHERE bss_id = @bss_id
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id)]
    )
    df = _bq.query(query, job_config=job_config).to_dataframe()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Piézomètre {bss_id} introuvable")

    row = df.iloc[0]
    if row[list(PERCENTILE_COLS.values())].isnull().any():
        raise HTTPException(status_code=422, detail=f"Seuils incomplets pour {bss_id}")

    return {key: float(row[col]) for key, col in PERCENTILE_COLS.items()}


# ══════════════════════════════════════════════════════════════════════════════
# PRÉCIPITATIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/piezo/{bss_id}/pluie")
def pluie(bss_id: str, days: int = 30):
    """Précipitations (pluie utile en mm) depuis chroniques_plean."""
    import pandas as pd
    query = f"""
        SELECT date_mesure, PU_synth
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET_ID}.chroniques_plean`
        WHERE bss_id = @bss_id
        ORDER BY date_mesure DESC
        LIMIT @days
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("bss_id", "STRING", bss_id),
            bigquery.ScalarQueryParameter("days", "INT64", days),
        ]
    )
    df = _bq.query(query, job_config=job_config).to_dataframe()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Aucune donnée de précipitation pour {bss_id}")
    df["date_mesure"] = pd.to_datetime(df["date_mesure"])
    df.sort_values("date_mesure", inplace=True)
    return {
        "bss_id": bss_id,
        "pluie": [
            {"date": row.date_mesure.strftime("%Y-%m-%d"), "pu_synth": round(float(row.PU_synth), 1)}
            for row in df.itertuples()
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRÉVISION
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/predict")
def predict(bss_id: str):
    """Prévision XGBoost autorégressive sur 13 semaines."""
    try:
        df_raw = load_plean(bss_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    df_w = preprocess_week(df_raw)

    FEATURE_COLS = ["semaine", "lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "moyenne_3w", "moyenne_6w"]
    X_train = df_w[FEATURE_COLS]
    y_train = df_w["niveau_nappe_eau"]
    model, _ = train(X_train, y_train, optimize=False)

    forecast = pred_future(model, df_w, n_weeks=13)

    return {
        "bss_id": bss_id,
        "prévision": [
            {"date": date.strftime("%Y-%m-%d"), "niveau": round(float(val), 3)}
            for date, val in forecast.items()
        ],
    }

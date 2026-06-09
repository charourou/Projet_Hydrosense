import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.preprocess.preprocessor import preprocess_week
from hydrosense.interface.main import preprocess, train, pred
from hydrosense.interface.main import train, pred_future

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return{
        'greeting': 'Hello'
    }

@app.get("/predict")
def predict(bss_id: str):

    # 1. Chargement + nettoyage
    df_raw = load_piezo_bq(bss_id)
    df_clean = clean_piezo(df_raw)

    # preprocess_week gère lui-même le set_index
    # 2. Feature engineering hebdomadaire
    df_w = preprocess_week(df_clean)

    # 3. Train sur toutes les données disponibles
    FEATURE_COLS = ["semaine", "lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "moyenne_3w", "moyenne_6w"]
    X_train = df_w[FEATURE_COLS]
    y_train = df_w["niveau_nappe_eau"]
    model, _ = train(X_train, y_train, optimize=False)

    # # 4. Prévision — 13 prochaines semaines (~90 jours)
    # X_future = X_train.tail(13).values
    # y_pred = model.predict(X_future)
    # last_date = df_w.index[-1]
    # forecast_index = pd.date_range(start=last_date, periods=14, freq="W")[1:]
    # forecast = pd.Series(y_pred, index=forecast_index)

    # /!\ peut etre probleme de data leakage avec pred_future ?
    # 4. Prévision autorégressive — 13 semaines futures
    forecast = pred_future(model, df_w, n_weeks=13)

    return {
        "bss_id": bss_id,
        "prévision": [
            {"date": date.strftime("%Y-%m-%d"), "niveau": round(float(val), 3)}
            for date, val in forecast.items()
        ],
    }

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hydrosense.database.bigquery import load_piezo_bq
from hydrosense.preprocess.cleaning import clean_piezo
from hydrosense.interface.main import preprocess, train, pred

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

    df_clean = df_clean.set_index("date_mesure")

    # 2. Feature engineering
    df_ml = preprocess(df_clean)

    # 3. Train
    X = df_ml[["mois", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]]
    y = df_ml["niveau_nappe_eau"]
    model, _ = train(X, y, optimize=False)

    # 4. Prévision
    X_future = df_ml[["mois", "lag_1", "lag_2", "lag_3", "lag_12", "moyenne_3m", "moyenne_6m"]].tail(3).values
    y_pred = model.predict(X_future)
    last_date = df_ml.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=4, freq="ME")[1:]
    forecast = pd.Series(y_pred, index=forecast_index)

    return {
        "bss_id": bss_id,
        "prévision": [
            {"date": date.strftime("%Y-%m-%d"), "niveau": round(float(val), 3)}
            for date, val in forecast.items()
        ],
    }
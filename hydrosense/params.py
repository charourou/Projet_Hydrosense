import os
import datetime

##################  VARIABLES  ##################
MODEL_VERSION=  os.environ.get("MODEL_VERSION", 'V2')
TEST_START_DATE= os.environ.get("TEST_START_DATE", 'V2')
MODEL_TARGET = ''
GCP_PROJECT_ID=os.environ.get("GCP_PROJECT_ID")
BQ_REGION=os.environ.get("BQ_REGION")
BQ_DATASET_ID=os.environ.get("BQ_DATASET_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")

MLFLOW_TRACKING_URI = ''
MLFLOW_EXPERIMENT = ''
MLFLOW_MODEL_NAME = ''
PREFECT_FLOW_NAME = ''
PREFECT_LOG_LEVEL = ''
EVALUATION_START_DATE = ''

##################  CONSTANTS  #####################



################  LISTE OF PIEZOS TARGETS ################
TARGETS_BSS = [
    "BSS002DEZW", # Bioule (Tarn-et-Garonne)
    "BSS000ZPHJ","BSS000ZQXN", # 35
    "BSS001PGUQ","BSS001QTKG","BSS001QSMT", "BSS001QHYH", # 79 dt Le Bourdet
    "BSS001QHPU","BSS001RQQE","BSS001SHNE", "BSS001VAJT", #Charente Maritime
    "BSS001UCZQ",
    "BSS001XBKZ", # 24
    "BSS001WVPW", "BSS002AFGV", "BSS002AXYY", "BSS002CAAM", "BSS002BGAF", #33 et 40 et 47
    "BSS002EDYK" # 64
]
# BSS002HNHX NaT NaT nan
# BSS002JYZT NaT NaT nan

install:
	pip install -r requirements.txt

run:
	streamlit run app/main.py

run_api:
	uvicorn hydrosense.api.fast:app --reload

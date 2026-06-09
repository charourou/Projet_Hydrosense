install:
	pip install -r requirements.txt

run:
	streamlit run app/main.py

run_api:
	uvicorn hydrosense.api.fast:app --reload

train:
	python hydrosense/interface/main.py

run_all:
	uvicorn hydrosense.api.fast:app --reload & streamlit run app/main.py
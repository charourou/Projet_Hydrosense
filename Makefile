install:
	pip install -r requirements.txt

reinstall_package:
	pip uninstall -y hydrosense || :
	pip install -e .

run:
	streamlit run app/main.py

run_api:
	uvicorn hydrosense.api.fast:app --reload

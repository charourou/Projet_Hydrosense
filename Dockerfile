FROM python:3.10.6
COPY hydrosense /hydrosense
COPY requirements-api.txt /requirements-api.txt
RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt
CMD uvicorn hydrosense.api.fast:app --host 0.0.0.0 --port ${PORT:-8080}

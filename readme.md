# Document Intelligence System — Streamlit UI

Frontend demo for the Document Intelligence System.
Calls the FastAPI backend deployed on GCP Cloud Run.

## Live Demo
https://amitrajput786-document-intelligence.streamlit.app

## Backend API
https://doc-intel-api-165472080585.asia-south1.run.app/docs

## What it does
Upload a document image (passport, degree, license) →
get an authenticity verdict with ELA map visualization,
confidence score, and extracted OCR text.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Full project (backend + Docker + GCP)
https://github.com/amitrajput786/document-intelligence-system
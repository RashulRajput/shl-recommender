# SHL Assessment Recommendation System

## Overview
Crawls SHL's Individual Test Solutions catalog and builds a hybrid recommender
to suggest relevant assessments for a given job description or query.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install -r requirements.txt

python crawler.py               # scrape SHL catalog
python build_index.py           # build models
uvicorn app:app --reload --port 8000

Test:
curl http://127.0.0.1:8000/health

Generate submission CSV:
python generate_predictions.py

The file predictions.csv will be created in the required format.
Deployment
Deploy easily on Render:


Build command: pip install -r requirements.txt


Start command: uvicorn app:app --host 0.0.0.0 --port $PORT


---

## ✅ RUN IN ORDER
```bash
python crawler.py          # builds data/assessments.csv
python build_index.py      # builds TF-IDF and embeddings
uvicorn app:app --reload   # start API
# In another terminal:
python generate_predictions.py  # builds predictions.csv (needs Gen_AI Dataset.xlsx or gen_ai_dataset.csv)
```

Once this is all saved, you can push it to GitHub and deploy to Render.

## Gen AI Query Dataset
Place your queries file as either:

- `data/Gen_AI Dataset.xlsx` (preferred Excel workbook; first column = queries)
- `data/gen_ai_dataset.csv` (CSV fallback; first column = queries)

The script `generate_predictions.py` will automatically pick the Excel file if present, else the CSV. It writes `predictions.csv` containing columns: Query, Assessment_url, Score.

If you want to regenerate the model index after changing assessments data, re-run:

```bash
python build_index.py
```

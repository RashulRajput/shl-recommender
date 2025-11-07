# SHL Assessment Recommendation Engine

FastAPI service that crawls SHL's Individual Test Solutions catalog, builds a hybrid semantic/TF-IDF index, and returns the best-fit assessment links for a hiring query. The pipeline also produces CSV submissions that meet SHL Gen-AI grader requirements.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Data Preparation](#data-preparation)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [API Usage](#api-usage)
- [Prediction Exports](#prediction-exports)
- [Testing](#testing)
- [Deployment (Render)](#deployment-render)
- [Troubleshooting](#troubleshooting)

## Features
- Web crawler with pre-packaged product filtering to keep only individual assessments.
- Hybrid recommender combining SentenceTransformer embeddings (60%) and TF-IDF cosine similarity (40%).
- Smart balancing that mixes knowledge ("K") and personality ("P") assessments when both technical and behavioral signals appear in the query.
- FastAPI endpoints for health checks and recommendations.
- Batch generator that produces both rich (`predictions.csv`) and submission-ready (`predictions_min.csv`) files.
- Validation script to guarantee the submission CSV matches grader expectations.

## Project Structure
```
SHL/
├── Procfile
├── .gitignore
└── shl/
		├── app.py
		├── build_index.py
		├── crawler.py
		├── generate_predictions.py
		├── requirements.txt
		├── README.md                # this file
		├── data/
		│   ├── assessments.csv
		│   └── gen_ai_dataset.csv   # or Gen_AI Dataset.xlsx
		├── models/
		│   ├── assessments_df.pkl
		│   ├── embedder.pkl
		│   ├── embeddings.npy
		│   ├── tfidf.pkl
		│   └── tfidf_matrix.pkl
		├── predictions.csv
		├── predictions_min.csv
		└── tests/
				├── check_submit.py
				└── test_api.sh
```

## Prerequisites
- Python 3.10+
- pip 22+
- (Optional) Git for version control

## Local Setup
```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Data Preparation
1. **Query Dataset**: place one of the following in `shl/data/`:
	 - `Gen_AI Dataset.xlsx`
	 - `gen_ai_dataset.csv`
	 Only the first column is read; each row should contain a job query.
2. **Assessments**: run the crawler to build `data/assessments.csv` (see pipeline below). You can also ship a pre-generated file when deploying.

## End-to-End Pipeline
Execute the following steps in order from inside `shl/` with the virtual environment activated:

```bash
python crawler.py                # scrape SHL catalog into data/assessments.csv
python build_index.py            # build TF-IDF + embedding artifacts in models/
python generate_predictions.py   # call API + produce predictions CSV files
```

To serve the API locally:
```bash
uvicorn app:app --reload --port 8000
```

## API Usage
### Health Check
```
GET /health  -> {"status": "ok"}
```

### Recommendation Endpoint
- **URL**: `POST /recommend`
- **Body**:
	```json
	{
		"job_title": "Java developer who collaborates with business teams",
		"description": "Optional additional context",
		"url": "Optional URL to scrape",
		"top_k": 5
	}
	```
- **Notes**:
	- `top_k` must be between 1 and 10.
	- When both technical and behavioral keywords are detected, at least one knowledge (K) and one personality (P) assessment are returned when available.

## Prediction Exports
- `predictions.csv`: columns `Query`, `Assessment_url`, `Score` for analysis.
- `predictions_min.csv`: two-column submission file consumed by the grader.

The script `python tests/check_submit.py` validates `predictions_min.csv` formatting (headers, row count, URL integrity).

## Testing
```bash
# submission CSV sanity checks
python tests/check_submit.py

# API smoke test (PowerShell example)
$body = '{"job_title":"Java developer","top_k":5}'
(Invoke-WebRequest -Uri "http://127.0.0.1:8000/recommend" -Method POST -Body $body -ContentType "application/json").Content

# run pytest (no unit tests yet, but keeps CI green)
python -m pytest
```

## Deployment (Render)
1. Push the repository to GitHub (Procfile already included).
2. In Render, create a **Web Service**:
	 - Environment: `Python`
	 - Build Command: `pip install -r requirements.txt`
	 - Start Command: `uvicorn shl.app:app --host 0.0.0.0 --port $PORT`
3. Ensure the `data/` and `models/` artifacts you need are committed or generated during build.

## Troubleshooting
- **`Method Not Allowed` when posting**: PowerShell's built-in `curl` defaults to GET. Use `Invoke-WebRequest` or `Invoke-RestMethod` with `-Method POST`.
- **`No module named uvicorn`**: install dependencies inside the activated virtual environment.
- **Port already in use**: stop existing Uvicorn instances (`Ctrl+C`) or choose another port via `--port`.

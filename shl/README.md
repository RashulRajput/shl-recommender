# SHL Assessment Recommendation System# SHL Assessment Recommendation System



This project is an intelligent Assessment Recommendation System, built according to the official SHL specification document and its provided dataset.This project crawls SHL’s catalog and recommends relevant assessments based on job descriptions.



It recommends the most relevant SHL assessments for a given job description using a hybrid approach that combines TF-IDF similarity and Gemini API–based semantic embeddings, delivered through a FastAPI backend.## Quick Start

```

## ⚙️ Tech Stack & Frameworkspython -m venv venv

venv\Scripts\activate

**Framework**: FastAPI (Python 3.10)pip install -r requirements.txt

python shl/build_index.py

**Libraries**: Scikit-learn, FAISS, Joblib, NumPy, Pandasuvicorn shl.app:app --reload --port 8000

```

**Embeddings Provider**: Gemini API (Google Generative AI)

### venv/ folder

**Deployment**: Render⚠️ Should NOT be in Git. Add `.venv/` (and `venv/`) to `.gitignore` before pushing.



**Environment Management**: venv## Table of Contents

- [Features](#features)

**API Testing**: PowerShell / curl- [Project Structure](#project-structure)

- [Prerequisites](#prerequisites)

## 🧩 System Overview- [Local Setup](#local-setup)

- [Data Preparation](#data-preparation)

**Crawler** (`crawler.py`) – Extracts assessment data from SHL's product catalogue.- [End-to-End Pipeline](#end-to-end-pipeline)

- [API Usage](#api-usage)

**Index Builder** (`build_index.py`) – Creates TF-IDF and embedding representations.- [Prediction Exports](#prediction-exports)

- [Testing](#testing)

**API** (`app.py`) – Provides endpoints:- [Deployment (Render)](#deployment-render)

- `/health` → Health check- [Troubleshooting](#troubleshooting)

- `/recommend` → Returns top-k matching assessments

## Features

**Prediction Generator** (`generate_predictions.py`) – Produces `predictions.csv` and `predictions_min.csv` in submission format.- Web crawler with pre-packaged product filtering to keep only individual assessments.

- Hybrid recommender combining SentenceTransformer embeddings (60%) and TF-IDF cosine similarity (40%).

## 📁 Directory Structure- Smart balancing that mixes knowledge ("K") and personality ("P") assessments when both technical and behavioral signals appear in the query.

```- FastAPI endpoints for health checks and recommendations.

SHL/- Batch generator that produces both rich (`predictions.csv`) and submission-ready (`predictions_min.csv`) files.

├── requirements.txt- Validation script to guarantee the submission CSV matches grader expectations.

├── Procfile

├── README.md## Project Structure

└── shl/```

    ├── app.pySHL/

    ├── crawler.py├── Procfile

    ├── build_index.py├── .gitignore

    ├── generate_predictions.py└── shl/

    ├── data/		├── app.py

    ├── models/		├── build_index.py

    │   ├── assessments_df.pkl		├── crawler.py

    │   ├── embeddings.npy		├── generate_predictions.py

    │   ├── tfidf.pkl		├── requirements.txt

    │   └── tfidf_matrix.pkl		├── README.md                # this file

    ├── predictions.csv		├── data/

    ├── predictions_min.csv		│   ├── assessments.csv

    └── tests/		│   └── gen_ai_dataset.csv   # or Gen_AI Dataset.xlsx

```		├── models/

		│   ├── assessments_df.pkl

## 🧪 How to Run		│   ├── embeddings.npy

		│   ├── tfidf.pkl

### 1️⃣ Setup		│   └── tfidf_matrix.pkl

```bash		├── predictions.csv

python -m venv venv		├── predictions_min.csv

venv\Scripts\activate		└── tests/

pip install -r requirements.txt				├── check_submit.py

```				└── test_api.sh

```

### 2️⃣ Build Index

```bash## Model Artifacts Policy

python shl/build_index.pyKeep these small runtime artifacts in Git (needed by `app.py`):

```- `shl/models/assessments_df.pkl`

- `shl/models/embeddings.npy`

### 3️⃣ Run Server- `shl/models/tfidf.pkl`

```powershell- `shl/models/tfidf_matrix.pkl`

$env:GEMINI_API_KEY="<your_gemini_api_key>"

$env:EMBED_API_PROVIDER="gemini"Do NOT commit heavy or unneeded artifacts:

uvicorn shl.app:app --port 8002- `shl/models/embedder.pkl` (remove; not needed if using Gemini/OpenAI at query time)

```- Any other large binaries: `*.pth`, `*.bin`, `*.pt`, `*.ckpt`, `*.onnx`, `*.safetensors`



### 4️⃣ Test EndpointsThese are already ignored via `.gitignore`.



**Health Check:**## Prerequisites

```powershell- Python 3.10+

Invoke-RestMethod -Uri "http://127.0.0.1:8002/health"- pip 22+

```- (Optional) Git for version control



**Recommendation Example:**## Local Setup

```powershell```bash

$body = '{"job_title":"Python developer who collaborates with backend teams","top_k":5}'# create and activate a virtual environment (Windows example)

Invoke-RestMethod -Uri "http://127.0.0.1:8002/recommend" -Method POST -Body $body -ContentType "application/json"python -m venv venv

```venv\Scripts\activate



## 📦 Output Example# install dependencies

```jsonpip install -r requirements.txt

{```

  "recommendations": [

    {## Data Preparation

      "assessment_name": "Coding Simulations",1. **Query Dataset**: place one of the following in `shl/data/`:

      "assessment_url": "https://www.shl.com/products/assessments/skills-and-simulations/coding-simulations/",	 - `Gen_AI Dataset.xlsx`

      "score": 0.34	 - `gen_ai_dataset.csv`

    }	 Only the first column is read; each row should contain a job query.

  ]2. **Assessments**: run the crawler to build `data/assessments.csv` (see pipeline below). You can also ship a pre-generated file when deploying.

}

```## End-to-End Pipeline

Execute the following steps in order from inside `shl/` with the virtual environment activated:

## 🌐 Deployment

```bash

**Build Command:**python crawler.py                # scrape SHL catalog into data/assessments.csv

```bashpython build_index.py            # build TF-IDF + embedding artifacts in models/

pip install -r requirements.txtpython generate_predictions.py   # call API + produce predictions CSV files

``````



**Start Command:**To serve the API locally:

```bash```bash

uvicorn shl.app:app --host 0.0.0.0 --port $PORTuvicorn shl.app:app --reload --port 8000

``````



**Environment Variables:**## API Usage

```### Health Check

GEMINI_API_KEY = <your_gemini_api_key>```

EMBED_API_PROVIDER = geminiGET /health  -> {"status": "ok"}

``````



## ✅ Completion Statement### Recommendation Endpoint

- **URL**: `POST /recommend`

I have successfully completed this Assessment Recommendation System according to the official SHL document and dataset requirements.- **Body**:

	```json

The project includes crawling, model building, hybrid recommendation logic, API endpoints, and deployment on Render.	{

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

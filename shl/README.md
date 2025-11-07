SHL Assessment Recommendation System

This project is an intelligent Assessment Recommendation System, built according to the official SHL specification document and its provided dataset.

It recommends the most relevant SHL assessments for a given job description using a hybrid approach that combines TF-IDF similarity and Gemini API–based semantic embeddings, delivered through a FastAPI backend.

⚙️ Tech Stack & Frameworks

Framework: FastAPI (Python 3.10)

Libraries: Scikit-learn, FAISS, Joblib, NumPy, Pandas

Embeddings Provider: Gemini API (Google Generative AI)

Deployment: Render

Environment Management: venv

API Testing: PowerShell / curl

🧩 System Overview

Crawler (crawler.py) – Extracts assessment data from SHL’s product catalogue.

Index Builder (build_index.py) – Creates TF-IDF and embedding representations.

API (app.py) – Provides endpoints:

/health → Health check

/recommend → Returns top-k matching assessments

Prediction Generator (generate_predictions.py) – Produces predictions.csv and predictions_min.csv in submission format.

📁 Directory Structure
SHL/
├── requirements.txt
├── Procfile
├── README.md
└── shl/
    ├── app.py
    ├── crawler.py
    ├── build_index.py
    ├── generate_predictions.py
    ├── data/
    ├── models/
    │   ├── assessments_df.pkl
    │   ├── embeddings.npy
    │   ├── tfidf.pkl
    │   └── tfidf_matrix.pkl
    ├── predictions.csv
    ├── predictions_min.csv
    └── tests/

🧪 How to Run
1️⃣ Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2️⃣ Build Index
python shl/build_index.py

3️⃣ Run Server
$env:GEMINI_API_KEY="<your_gemini_api_key>"
$env:EMBED_API_PROVIDER="gemini"
uvicorn shl.app:app --port 8002

4️⃣ Test Endpoints

Health Check:

Invoke-RestMethod -Uri "http://127.0.0.1:8002/health"


Recommendation Example:

$body = '{"job_title":"Python developer who collaborates with backend teams","top_k":5}'
Invoke-RestMethod -Uri "http://127.0.0.1:8002/recommend" -Method POST -Body $body -ContentType "application/json"

📦 Output Example
{
  "recommendations": [
    {
      "assessment_name": "Coding Simulations",
      "assessment_url": "https://www.shl.com/products/assessments/skills-and-simulations/coding-simulations/",
      "score": 0.34
    }
  ]
}

🌐 Deployment

Build Command:

pip install -r requirements.txt


Start Command:

uvicorn shl.app:app --host 0.0.0.0 --port $PORT


Environment Variables:

GEMINI_API_KEY = <your_gemini_api_key>
EMBED_API_PROVIDER = gemini

✅ Completion Statement

I have successfully completed this Assessment Recommendation System according to the official SHL document and dataset requirements.
The project includes crawling, model building, hybrid recommendation logic, API endpoints, and deployment on Render.

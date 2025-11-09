# SHL Assessment Recommender System

AI-powered recommendation engine for SHL assessments using hybrid TF-IDF + Gemini embeddings.

## 📊 System Overview

- **407 assessments** covering Technical, Behavioural, Cognitive, Simulation, and General categories
- **Hybrid scoring**: Combines TF-IDF lexical matching (40%) + Gemini semantic embeddings (60%)
- **FastAPI server** with interactive web UI
- **Training data**: 10 labeled queries, 9 test queries for evaluation

## 🏗️ Project Structure

```
SHL/
├── shl/
│   ├── app.py                          # FastAPI recommendation server
│   ├── crawler.py                      # SHL product catalog crawler
│   ├── build_index.py                  # Build TF-IDF + embeddings index
│   ├── embed_gemini_only.py           # Gemini embedding wrapper (with fallbacks)
│   ├── generate_predictions.py         # Generate predictions for test queries
│   ├── data/
│   │   ├── assessments_full.csv       # Complete SHL catalog (407 assessments)
│   │   ├── assessments.csv            # Active catalog used by recommender
│   │   ├── Gen_AI_Train-Set_FULL.csv  # Training data (10 queries, 65 examples)
│   │   └── Gen_AI_Test-Set_FULL.csv   # Test data (9 unlabeled queries)
│   ├── models/
│   │   ├── assessments_df.pkl         # Processed assessments DataFrame
│   │   ├── tfidf.pkl                  # TF-IDF vectorizer
│   │   ├── tfidf_matrix.pkl           # TF-IDF document matrix
│   │   └── embeddings.npy             # Gemini embeddings (optional)
│   └── scripts/
│       ├── fix_assessments.py         # CSV cleaning utilities
│       └── verify_crawl.py            # Validation script for catalog
├── requirements.txt
├── Procfile                            # Render deployment config
└── test_recommendations.py             # Testing script
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

**Required packages:**
- fastapi, uvicorn (web server)
- requests, beautifulsoup4 (crawler)
- pandas, numpy, scikit-learn (ML)
- google-generativeai (embeddings)

### 2. Set Environment Variables

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
$env:GEMINI_EMBED_MODEL = "textembedding-gecko-001"  # optional
```

### 3. Crawl SHL Catalog (Optional - already done)

```powershell
cd shl
python crawler.py
```

**Output:** `shl/data/assessments_full.csv` (407 assessments)

**Verify crawl quality:**
```powershell
python scripts/verify_crawl.py data/assessments_full.csv
```

### 4. Build Search Index

```powershell
# Copy full catalog to active file
Copy-Item shl/data/assessments_full.csv shl/data/assessments.csv

# Build TF-IDF + embeddings
cd shl
python build_index.py
```

**Generates:**
- `models/assessments_df.pkl` - Processed data
- `models/tfidf.pkl`, `models/tfidf_matrix.pkl` - TF-IDF index
- `models/embeddings.npy` - Gemini embeddings (or fallback pseudo-embeddings)

### 5. Start Recommendation Server

```powershell
cd shl
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

**Access:**
- Web UI: http://localhost:8002
- API health: http://localhost:8002/health
- API docs: http://localhost:8002/docs

## 📡 API Usage

### POST /recommend

**Request:**
```json
{
  "job_title": "Senior Java Developer with strong collaboration skills",
  "description": "",
  "url": "",
  "top_k": 5
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "assessment_name": "Core Java (Advanced Level) (New)",
      "assessment_url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "score": 0.92
    },
    {
      "assessment_name": "Interpersonal Communications",
      "assessment_url": "https://www.shl.com/products/product-catalog/view/interpersonal-communications/",
      "score": 0.87
    }
  ]
}
```

## 🔧 Maintenance

### Re-crawl Catalog

When SHL adds new assessments:

```powershell
cd shl
python crawler.py                                   # Crawl latest catalog
python scripts/verify_crawl.py data/assessments_full.csv  # Verify
Copy-Item data/assessments_full.csv data/assessments.csv  # Activate
python build_index.py                               # Rebuild index
```

### Update Embeddings Only

If catalog structure unchanged but want fresh embeddings:

```powershell
Remove-Item shl/models/embeddings.npy
cd shl
python build_index.py
```

## 🧪 Testing

### Test Recommendations

```powershell
python test_recommendations.py
```

### Generate Predictions for Test Set

```powershell
cd shl
python generate_predictions.py
```

**Output:** Predictions for 9 test queries based on training data similarity

## 📊 Catalog Statistics

- **Total Assessments:** 407
- **Test Type Distribution:**
  - General: 223 (54.8%)
  - Behavioural: 69 (17.0%)
  - Technical: 63 (15.5%)
  - Simulation: 35 (8.6%)
  - Cognitive: 17 (4.2%)

**Key Technical Assessments Included:**
- Core Java (Entry Level & Advanced Level)
- Python, JavaScript, C++, C#
- SQL, SQL Server, MySQL, Oracle
- Selenium, Automata coding simulations
- React, Angular, Node.js, Spring, Hibernate
- Docker, Kubernetes, AWS, Azure

**Key Behavioural Assessments:**
- OPQ (Occupational Personality Questionnaire)
- MQ (Motivation Questionnaire)
- Interpersonal Communications
- Leadership, Teamwork, Situational Judgment

## 🎯 Training Data

**Training Set:** 10 queries with human-labeled relevant assessments
- Java developers + collaboration → Core Java, Interpersonal Communications
- QA Engineer → Selenium, SQL Server, JavaScript, Automata SQL
- Data Analyst → Python, SQL, Tableau, Excel

**Test Set:** 9 unlabeled queries requiring predictions
- Used for challenge/competition evaluation

## 🔬 Algorithm

### Hybrid Scoring

```python
final_score = 0.6 × embedding_similarity + 0.4 × tfidf_similarity
```

**Features:**
- **TF-IDF**: Exact keyword matching (technical terms, role names)
- **Embeddings**: Semantic understanding (synonyms, context)
- **Domain boost**: +0.08 for technical assessments when query mentions coding/programming
- **K/P balancing**: Ensures mix of Knowledge (K) and Personality (P) tests when query indicates both needs

### Test Type Classification

Regex-based heuristics classify assessments:
- **Technical**: Java, Python, SQL, coding, programming, developer
- **Behavioural**: Personality, OPQ, motivation, communication, teamwork
- **Cognitive**: Reasoning, numerical, verbal, aptitude, ability
- **Simulation**: Call center, sales simulation, role play
- **General**: Default category

## 🚢 Deployment (Render)

**Environment Variables:**
```
GEMINI_API_KEY=<your-key>
EMBED_API_PROVIDER=gemini
GEMINI_EMBED_MODEL=textembedding-gecko-001
```

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
cd shl && uvicorn app:app --host 0.0.0.0 --port $PORT
```

## 📝 Notes

- **Embeddings fallback:** If Gemini API fails, system uses existing embeddings or generates deterministic pseudo-embeddings (384-dim) for safe operation
- **Throttling:** Crawler uses 0.7s delay between requests to respect SHL servers
- **robots.txt:** Crawler checks and respects robots.txt directives
- **Missing test_type:** Automatically filled with 'general' during index build

## 👤 Author

Rashul Rajput

## 📄 License

All rights reserved. SHL assessment content © SHL and its affiliates.

# SHL Recommender - Complete Pipeline Guide

**Step-by-step guide to rebuild the full recommendation system from scratch.**

## Prerequisites

```powershell
# Activate virtual environment (if using)
& .\venv\Scripts\Activate.ps1

# Or ensure packages are installed
pip install -r requirements.txt
```

**Required environment variable:**
```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
$env:EMBED_API_PROVIDER = "gemini"
```

---

## Step 1: Crawl Full SHL Catalog (32 pages → 400+ assessments)

```powershell
# Run crawler to fetch all individual test solutions
python .\shl\crawler.py --output shl\data\assessments_full.csv --pages 32
```

**Expected output:**
- `shl/data/assessments_full.csv` with ~407 assessments
- Columns: `title`, `url`, `test_type`, `raw_text`, `text`
- Duration: ~5-10 minutes (0.7s throttle per request)

**What it does:**
- Scans 32 catalog pages: `?start=0`, `?start=12`, ..., `?start=372` with `&type=1`
- Extracts product links from each catalog page
- Visits each product page to get title, description, and URL
- Classifies test type: technical, behavioural, cognitive, simulation, general
- Skips "pre-packaged job solutions"

---

## Step 2: Verify Crawled Data Quality

```powershell
# Quick verification check
python -c "import pandas as pd; df = pd.read_csv('shl/data/assessments_full.csv'); print('Rows:', len(df)); print('Columns:', df.columns.tolist()); print('\nSample:'); print(df[['title','url']].head(10)); numeric = df['title'].astype(str).str.match(r'^\d+$', na=False).sum(); print('\nNumeric-only titles:', numeric)"
```

**Or use the verification script:**
```powershell
python .\shl\scripts\verify_crawl.py shl\data\assessments_full.csv
```

**Expected checks:**
- ✅ ~407 assessments total
- ✅ All required columns present: `title`, `url`, `test_type`, `raw_text`, `text`
- ✅ No numeric-only titles
- ✅ Key assessments present: Core Java, Python, SQL, Selenium, Automata, Interpersonal Communications
- ✅ No duplicate URLs
- ✅ Good content quality (descriptions > 100 chars)

**If numeric-only titles exist:**
```powershell
python .\shl\scripts\fix_assessments.py shl\data\assessments_full.csv
```

---

## Step 3: Activate the Full Catalog

```powershell
# Copy full catalog to active filename used by build_index.py
Copy-Item shl\data\assessments_full.csv shl\data\assessments.csv -Force
```

**Why:** `build_index.py` reads from `data/assessments.csv`. Keeping both files allows you to:
- Preserve original crawl: `assessments_full.csv`
- Use active catalog: `assessments.csv` (can be filtered/modified)

---

## Step 4: Build TF-IDF + Embeddings + Index

```powershell
# Ensure API key is set
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
$env:EMBED_API_PROVIDER = "gemini"

# Remove old models to force fresh rebuild
Remove-Item shl\models\*.pkl -Force
Remove-Item shl\models\embeddings.npy -Force

# Build index
cd shl
python build_index.py
cd ..
```

**Generated files:**
- `shl/models/assessments_df.pkl` - Processed DataFrame (407 rows)
- `shl/models/tfidf.pkl` - TF-IDF vectorizer
- `shl/models/tfidf_matrix.pkl` - TF-IDF document matrix (407, ~2800)
- `shl/models/embeddings.npy` - Gemini embeddings (407, 384)

**What it does:**
1. Loads `data/assessments.csv`
2. Fills missing `test_type` with 'general'
3. Loads training/test data mappings (Gen_AI_Train-Set_FULL.csv, Gen_AI_Test-Set_FULL.csv)
4. Builds TF-IDF matrix with n-grams (1,3)
5. Generates embeddings via Gemini API (or pseudo-embeddings if API fails)
6. Saves all artifacts to `models/` directory

**Fallback behavior:**
- If Gemini API fails → generates deterministic pseudo-embeddings (384-dim)
- System still works with TF-IDF alone (40% of scoring)

---

## Step 5: Verify Models Are Aligned

```powershell
# Check all models have same number of rows
python -c "import joblib, numpy as np; df = joblib.load('shl/models/assessments_df.pkl'); emb = np.load('shl/models/embeddings.npy'); tfidf = joblib.load('shl/models/tfidf_matrix.pkl'); print('Rows in DF:', len(df)); print('Embedding shape:', emb.shape); print('TF-IDF shape:', tfidf.shape); assert len(df) == emb.shape[0] == tfidf.shape[0], 'MISMATCH!'; print('All aligned ✅')"
```

**Expected output:**
```
Rows in DF: 407
Embedding shape: (407, 384)
TF-IDF shape: (407, 2832)
All aligned ✅
```

---

## Step 6: Start Recommendation Server

```powershell
# Set API key (if not already set)
$env:GEMINI_API_KEY = "your-gemini-api-key-here"

# Start server
cd shl
python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

**Or from root directory:**
```powershell
python -m uvicorn shl.app:app --host 0.0.0.0 --port 8002 --reload
```

**Access points:**
- Web UI: http://localhost:8002
- Health check: http://localhost:8002/health
- API docs: http://localhost:8002/docs

---

## Step 7: Test Recommendations

### Via PowerShell (REST API)

**Health check:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8002/health"
```

**Test recommendation:**
```powershell
$body = @{
    job_title = "I am hiring for Java developers who can also collaborate effectively with my business teams"
    description = ""
    url = ""
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8002/recommend" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

**Expected results:**
1. Core Java (Advanced Level) - technical
2. Core Java (Entry Level) - technical
3. Interpersonal Communications - behavioural
4. Java-related (Spring, Hibernate, Design Patterns)
5. Communication/collaboration assessments

### Via Web UI

1. Open http://localhost:8002 in browser
2. Enter query: "I am hiring for Java developers who can also collaborate effectively with my business teams"
3. Set Top K: 5
4. Click "Recommend"

### Via Python Test Script

```powershell
python test_recommendations.py
```

---

## Step 8: Generate Predictions for Test Set (Optional)

```powershell
cd shl
python generate_predictions.py
```

**Output:** `predictions.csv` with recommendations for 9 test queries based on training data similarity.

---

## Troubleshooting

### Issue: "Gemini API variant failed"
**Solution:** System automatically falls back to pseudo-embeddings. TF-IDF still works (40% of scoring).

To get real embeddings:
```powershell
# Try updating google-generativeai
pip install --upgrade google-generativeai

# Or use a different embedding model
$env:GEMINI_EMBED_MODEL = "models/embedding-001"
```

### Issue: "Row count mismatch"
**Solution:** Delete old models and rebuild from scratch:
```powershell
Remove-Item shl\models\* -Force
cd shl
python build_index.py
```

### Issue: "Empty recommendations"
**Solution:** Check models loaded correctly:
```powershell
# Check app.py startup logs
# Should see: "models loaded successfully with embeddings" or "TF-IDF only mode"
```

### Issue: "Crawler returns few assessments"
**Solution:** Increase pages or check network:
```powershell
python .\shl\crawler.py --pages 50  # Try more pages
```

---

## Complete Reset (Nuclear Option)

```powershell
# Delete everything and start fresh
Remove-Item shl\data\assessments*.csv -Force
Remove-Item shl\models\* -Force

# Re-run full pipeline
python .\shl\crawler.py --output shl\data\assessments_full.csv --pages 32
python .\shl\scripts\verify_crawl.py shl\data\assessments_full.csv
Copy-Item shl\data\assessments_full.csv shl\data\assessments.csv -Force
cd shl
python build_index.py
python -m uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

---

## Deployment Checklist

✅ Environment variables set (GEMINI_API_KEY, EMBED_API_PROVIDER)  
✅ `shl/data/assessments.csv` exists with 407 assessments  
✅ All models in `shl/models/` directory:
  - assessments_df.pkl
  - tfidf.pkl, tfidf_matrix.pkl
  - embeddings.npy  
✅ Training/test data present:
  - Gen_AI_Train-Set_FULL.csv
  - Gen_AI_Test-Set_FULL.csv  
✅ Server starts without errors  
✅ Health check returns `{"status":"ok"}`  
✅ Test query returns relevant assessments  

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python shl/crawler.py --pages 32` | Crawl full catalog |
| `python shl/scripts/verify_crawl.py shl/data/assessments_full.csv` | Verify data |
| `python shl/build_index.py` | Build search index |
| `python -m uvicorn shl.app:app --port 8002` | Start server |
| `Invoke-RestMethod http://localhost:8002/health` | Health check |

---

## Performance Expectations

- **Crawling:** 5-10 minutes for 32 pages (407 assessments)
- **Index building:** 2-5 minutes (depends on Gemini API)
- **Server startup:** < 5 seconds
- **Recommendation response:** < 1 second per query
- **Memory usage:** ~500MB for models + server

---

## File Size Reference

- `assessments_full.csv`: ~800 KB (407 rows)
- `assessments_df.pkl`: ~800 KB
- `tfidf_matrix.pkl`: ~485 KB
- `embeddings.npy`: ~625 KB (407×384 float32)
- Total models: ~2 MB

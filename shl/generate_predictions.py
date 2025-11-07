import pandas as pd
import requests
import os

LOCAL_URL = "http://127.0.0.1:8000/recommend"
DEFAULT_XLSX = "data/Gen_AI Dataset.xlsx"
ALT_CSV = "data/gen_ai_dataset.csv"

def load_queries():
	"""Load first column of Gen AI dataset from XLSX (preferred) or CSV fallback."""
	# Prefer CSV if present (explicit user-provided query list), else fallback to Excel.
	if os.path.exists(ALT_CSV):
		df = pd.read_csv(ALT_CSV)
		print(f"Loaded {ALT_CSV} with {len(df)} rows")
	elif os.path.exists(DEFAULT_XLSX):
		df = pd.read_excel(DEFAULT_XLSX)
		print(f"Loaded {DEFAULT_XLSX} with {len(df)} rows")
	else:
		raise FileNotFoundError(f"Place your dataset at '{DEFAULT_XLSX}' or '{ALT_CSV}'.")
	if df.empty:
		raise ValueError("Dataset is empty.")
	first_col = df.columns[0]
	return df[first_col].astype(str).tolist()

def call_api(q):
	payload = {"job_title": q, "top_k": 10}
	r = requests.post(LOCAL_URL, json=payload)
	r.raise_for_status()
	return r.json()

def main():
    queries = load_queries()
    rows = []
    for q in queries:
        print("Processing:", q)
        try:
            resp = call_api(q)
        except Exception as e:
            print(f"API error for '{q}': {e}")
            continue
        recs = resp.get("recommendations", [])
        for r in recs:
            rows.append((q, r["assessment_url"], r.get("score")))
    out = pd.DataFrame(rows, columns=["Query","Assessment_url","Score"])
    out.to_csv("predictions.csv", index=False)
    print(f"Saved predictions.csv with {len(out)} recommendation rows")
    
    # Export grader-ready two-column CSV
    out[['Query','Assessment_url']].to_csv('predictions_min.csv', index=False)
    print(f"Saved predictions_min.csv (submission-ready) with {len(out)} rows")

if __name__ == "__main__":
    main()
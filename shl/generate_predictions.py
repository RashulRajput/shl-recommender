import pandas as pd
import requests
import os

LOCAL_URL = "http://127.0.0.1:8002/recommend"
TEST_FILE = "data/Gen_AI_Test-Set_FULL.csv"

def load_queries():
	"""Load test queries from Gen_AI_Test-Set_FULL.csv"""
	if not os.path.exists(TEST_FILE):
		raise FileNotFoundError(f"Test file not found: '{TEST_FILE}'")
	
	df = pd.read_csv(TEST_FILE)
	print(f"Loaded {TEST_FILE} with {len(df)} rows")
	
	if df.empty:
		raise ValueError("Dataset is empty.")
	
	# Get the Query column
	if 'Query' in df.columns:
		return df['Query'].astype(str).tolist()
	else:
		# Use first column if Query not found
		first_col = df.columns[0]
		return df[first_col].astype(str).tolist()

def call_api(q):
	payload = {"job_title": q, "description": "", "top_k": 10}
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
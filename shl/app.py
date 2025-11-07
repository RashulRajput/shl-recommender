from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd, requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

app = FastAPI()

class Req(BaseModel):
    job_title: str = ""
    description: str = ""
    url: str = ""
    top_k: int = 5

df = joblib.load("models/assessments_df.pkl")
embeddings = np.load("models/embeddings.npy")
embedder = joblib.load("models/embedder.pkl")
tf = joblib.load("models/tfidf.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

def fetch_url_text(u):
	try:
		r = requests.get(u, timeout=10)
		soup = BeautifulSoup(r.text, "html.parser")
		for s in soup(["script","style","noscript"]): s.decompose()
		return soup.get_text(separator=" ", strip=True)
	except Exception: return ""

def normalize(x):
	x = np.array(x, dtype=float)
	return (x - x.min()) / (x.max() - x.min() + 1e-9)

@app.get("/health")
def health():
	return {"status":"ok"}

@app.post("/recommend")
def recommend(req: Req):
    try:
        if not any([req.job_title, req.description, req.url]):
            raise HTTPException(status_code=400, detail="No input provided")
        if req.top_k < 1 or req.top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

        q_text = " ".join([req.job_title or "", req.description or ""])
        if req.url:
            q_text += " " + fetch_url_text(req.url)

        q_emb = embedder.encode([q_text], convert_to_numpy=True)
        sem = cosine_similarity(q_emb, embeddings)[0]
        sem_n = normalize(sem)
        q_vec = tf.transform([q_text])
        tfidf_sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        tf_n = normalize(tfidf_sims)
        final = 0.6 * sem_n + 0.4 * tf_n

        # Balancing: detect technical & behavioural tokens
        tech_tokens = ["python","java","sql","javascript","react","node","c++","c#","docker","kubernetes","coding","technical","software","engineer"]
        beh_tokens = ["collaborat","team","stakehold","communication","personality","behaviour","behavior","leadership","culture","interpersonal"]
        txt_low = q_text.lower()
        has_tech = any(t in txt_low for t in tech_tokens)
        has_beh = any(t in txt_low for t in beh_tokens)

        idx_sorted = np.argsort(final)[::-1]
        top_k = req.top_k

        # If both domains requested, enforce at least one K and one P if available
        selected_idx = []
        if has_tech and has_beh and top_k >= 2:
            need_k = max(1, top_k // 2)
            need_p = max(1, top_k - need_k)
            k_added = 0
            p_added = 0
            test_types = df.get("test_type", pd.Series([""]*len(df))).fillna("").astype(str).tolist()
            for i in idx_sorted:
                if len(selected_idx) >= top_k:
                    break
                t = test_types[i].upper() if i < len(test_types) else ""
                # treat presence of 'K' as knowledge, 'P' as personality
                if "K" in t and k_added < need_k:
                    selected_idx.append(i); k_added += 1
                elif "P" in t and p_added < need_p:
                    selected_idx.append(i); p_added += 1
            # fill any remaining slots with top-ranked
            for i in idx_sorted:
                if i not in selected_idx and len(selected_idx) < top_k:
                    selected_idx.append(i)
        else:
            selected_idx = list(idx_sorted[:top_k])

        out = []
        for i in selected_idx:
            out.append({
                "assessment_name": df.iloc[i]["title"],
                "assessment_url": df.iloc[i]["url"],
                "score": float(final[i])
            })
        return {"recommendations": out}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
import os

import google.generativeai as genai
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="SHL Assessment Recommender")


class Req(BaseModel):
    job_title: str = ""
    description: str = ""
    url: str = ""
    top_k: int = 5


EMBED_PROVIDER = os.getenv("EMBED_API_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "textembedding-gecko-001")
MODELS_READY = False
df = None
embeddings = None
tf = None
tfidf_matrix = None


def load_models():
    """
    Load only precomputed artifacts: assessments_df.pkl, embeddings.npy, tfidf.pkl, tfidf_matrix.pkl
    """
    global MODELS_READY, df, embeddings, tf, tfidf_matrix
    try:
        required = [
            "shl/models/assessments_df.pkl",
            "shl/models/embeddings.npy",
            "shl/models/tfidf.pkl",
            "shl/models/tfidf_matrix.pkl",
        ]
        for p in required:
            if not os.path.exists(p):
                MODELS_READY = False
                print(f"[load_models] missing: {p}")
                return

        df = joblib.load("shl/models/assessments_df.pkl")
        embeddings = np.load("shl/models/embeddings.npy")
        tf = joblib.load("shl/models/tfidf.pkl")
        tfidf_matrix = joblib.load("shl/models/tfidf_matrix.pkl")
        MODELS_READY = True
        print("[load_models] models loaded successfully")
    except Exception as e:
        MODELS_READY = False
        print("[load_models] ERROR loading models:", str(e))


@app.on_event("startup")
def on_startup():
    # configure Gemini
    if EMBED_PROVIDER == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            print("[startup] Gemini configured")
        else:
            print("[startup] GEMINI_API_KEY not set")
    load_models()


def fetch_url_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def normalize(values):
    array = np.array(values, dtype=float)
    return (array - array.min()) / (array.max() - array.min() + 1e-9)


def get_query_embedding_gemini(text: str) -> np.ndarray:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    # create embedding
    resp = genai.embeddings.create(model=GEMINI_EMBED_MODEL, input=text)
    # response structure: resp.data[0].embedding
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    return emb


def get_query_embedding(text: str) -> np.ndarray:
    if EMBED_PROVIDER == "gemini":
        return get_query_embedding_gemini(text)
    else:
        raise RuntimeError("Unsupported EMBED_PROVIDER. Use 'gemini'.")


@app.get("/health")
def health():
    status = "ok" if MODELS_READY else "starting"
    return {"status": status}


@app.post("/recommend")
def recommend(req: Req):
    try:
        if not any([req.job_title, req.description, req.url]):
            raise HTTPException(status_code=400, detail="No input provided")
        if req.top_k < 1 or req.top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
        if not MODELS_READY:
            raise HTTPException(status_code=503, detail="Models not ready yet. Please try again shortly.")

        # Build query text
        q_text = " ".join([req.job_title or "", req.description or ""]).strip()
        if req.url:
            q_text += " " + fetch_url_text(req.url)

        # 1) Query embedding via Gemini
        try:
            q_emb = get_query_embedding(q_text)
        except Exception as e:
            # If external API fails, fallback to TF-IDF only
            print("[recommend] embedding API failed:", str(e))
            q_emb = None

        # 2) Semantic similarity (if q_emb present)
        if q_emb is not None:
            sem = cosine_similarity(q_emb.reshape(1, -1), embeddings)[0]
            sem_n = normalize(sem)
        else:
            sem_n = np.zeros(len(df))

        # 3) TF-IDF similarity
        q_vec = tf.transform([q_text])
        tfidf_sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        tf_n = normalize(tfidf_sims)

        # 4) Combine hybrid score
        # adjust weights as needed
        final = 0.65 * sem_n + 0.35 * tf_n

        # 5) Balance K (Knowledge) and P (Personality) if both domains present
        tech_tokens = ["python","java","sql","javascript","react","node","c++","c#","docker","kubernetes","coding","technical","software","engineer","qa","selenium"]
        beh_tokens = ["collaborat","team","stakehold","communication","personality","behaviour","behavior","leadership","culture","interpersonal"]
        txt_low = q_text.lower()
        has_tech = any(t in txt_low for t in tech_tokens)
        has_beh = any(t in txt_low for t in beh_tokens)

        idx_sorted = np.argsort(final)[::-1]
        top_k = req.top_k

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
                if "K" in t and k_added < need_k:
                    selected_idx.append(i); k_added += 1
                elif "P" in t and p_added < need_p:
                    selected_idx.append(i); p_added += 1
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
df = joblib.load("models/assessments_df.pkl")
tf = joblib.load("models/tfidf.pkl")
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


app = FastAPI()


class Req(BaseModel):
    job_title: str = ""
    description: str = ""
    url: str = ""
    top_k: int = 5


EMBED_PROVIDER = os.getenv("EMBED_API_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODELS_READY = False
df = None
embeddings = None
tf = None
tfidf_matrix = None


def load_models():
    global MODELS_READY, df, embeddings, tf, tfidf_matrix
    required_paths = [
        "models/assessments_df.pkl",
        "models/embeddings.npy",
        "models/tfidf.pkl",
        "models/tfidf_matrix.pkl",
    ]

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        MODELS_READY = False
        print(f"[load_models] missing artifacts: {missing}")
        return

    try:
        df = joblib.load("models/assessments_df.pkl")
        embeddings = np.load("models/embeddings.npy")
        tf = joblib.load("models/tfidf.pkl")
        tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
        MODELS_READY = True
        print("[load_models] model artifacts loaded")
    except Exception as exc:
        MODELS_READY = False
        print(f"[load_models] failed: {exc}")


@app.on_event("startup")
def on_startup():
    if EMBED_PROVIDER == "gemini":
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            print("[startup] GEMINI_API_KEY not provided; embeddings will fail")
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
        raise RuntimeError("GEMINI_API_KEY not configured")

    model_name = os.getenv("GEMINI_EMBED_MODEL", "textembedding-gecko-001")
    response = genai.embeddings.create(model=model_name, input=text)
    vector = np.array(response.data[0].embedding, dtype=np.float32)
    return vector


def get_query_embedding(text: str) -> np.ndarray:
    if EMBED_PROVIDER == "gemini":
        return get_query_embedding_gemini(text)
    raise RuntimeError("Unsupported EMBED_API_PROVIDER; expected 'gemini'")


@app.get("/health")
def health():
    status = "ok" if MODELS_READY else "loading"
    return {"status": status}


@app.post("/recommend")
def recommend(req: Req):
    try:
        if not MODELS_READY:
            raise HTTPException(status_code=503, detail="Model artifacts not loaded")
        if not any([req.job_title, req.description, req.url]):
            raise HTTPException(status_code=400, detail="No input provided")
        if req.top_k < 1 or req.top_k > 10:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

        q_text_parts = [req.job_title or "", req.description or ""]
        q_text = " ".join(part for part in q_text_parts if part).strip()

        if req.url:
            q_text += " " + fetch_url_text(req.url)

        q_text = q_text.strip()
        if not q_text:
            raise HTTPException(status_code=400, detail="Query text is empty")

        q_emb = get_query_embedding(q_text).reshape(1, -1)
        sem = cosine_similarity(q_emb, embeddings)[0]
        sem_n = normalize(sem)

        q_vec = tf.transform([q_text])
        tfidf_sims = cosine_similarity(q_vec, tfidf_matrix)[0]
        tf_n = normalize(tfidf_sims)
        final = 0.6 * sem_n + 0.4 * tf_n

        tech_tokens = [
            "python",
            "java",
            "sql",
            "javascript",
            "react",
            "node",
            "c++",
            "c#",
            "docker",
            "kubernetes",
            "coding",
            "technical",
            "software",
            "engineer",
        ]
        beh_tokens = [
            "collaborat",
            "team",
            "stakehold",
            "communication",
            "personality",
            "behaviour",
            "behavior",
            "leadership",
            "culture",
            "interpersonal",
        ]

        txt_low = q_text.lower()
        has_tech = any(token in txt_low for token in tech_tokens)
        has_beh = any(token in txt_low for token in beh_tokens)

        idx_sorted = np.argsort(final)[::-1]
        top_k = req.top_k

        selected_idx = []
        if has_tech and has_beh and top_k >= 2:
            need_k = max(1, top_k // 2)
            need_p = max(1, top_k - need_k)
            k_added = 0
            p_added = 0
            test_types = (
                df.get("test_type", pd.Series([""] * len(df)))
                .fillna("")
                .astype(str)
                .tolist()
            )
            for idx in idx_sorted:
                if len(selected_idx) >= top_k:
                    break
                label = test_types[idx].upper() if idx < len(test_types) else ""
                if "K" in label and k_added < need_k:
                    selected_idx.append(idx)
                    k_added += 1
                elif "P" in label and p_added < need_p:
                    selected_idx.append(idx)
                    p_added += 1
            for idx in idx_sorted:
                if idx not in selected_idx and len(selected_idx) < top_k:
                    selected_idx.append(idx)
        else:
            selected_idx = list(idx_sorted[:top_k])

        recommendations = []
        for idx in selected_idx:
            row = df.iloc[idx]
            recommendations.append(
                {
                    "assessment_name": row["title"],
                    "assessment_url": row["url"],
                    "score": float(final[idx]),
                }
            )
        return {"recommendations": recommendations}
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
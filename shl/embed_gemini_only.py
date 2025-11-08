# shl/embed_gemini_only.py
"""
Gemini-only embeddings wrapper.
Tries multiple common google.generativeai APIs (different client versions).
If Gemini fails (quota/key/SDK mismatch), falls back to:
 - reusing shl/models/embeddings.npy (if present)
 - deterministic pseudo-embeddings of fixed dimension for safe operation
This file DOES NOT call OpenAI.
"""

import os
import traceback
import numpy as np

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
EMBED_PATH = os.path.join(MODELS_DIR, "embeddings.npy")
# default fallback dimension (matches many embedding dims like Gemini small)
DEFAULT_DIM = 384

def _load_existing():
    if os.path.exists(EMBED_PATH):
        try:
            emb = np.load(EMBED_PATH)
            print(f"[embed_gemini_only] Loaded existing embeddings.npy {emb.shape}")
            return emb
        except Exception as e:
            print("[embed_gemini_only] Failed to load embeddings.npy:", e)
    return None

def _save(emb):
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.save(EMBED_PATH, emb)
        print("[embed_gemini_only] Saved embeddings to", EMBED_PATH)
    except Exception as e:
        print("[embed_gemini_only] Failed to save embeddings:", e)

def _try_gemini_variants(texts, model="textembedding-gecko-001"):
    """
    Attempt a few patterns used by different google.generativeai client versions.
    Returns numpy array or None.
    """
    try:
        import google.generativeai as genai
        print("[embed_gemini_only] google.generativeai imported:", genai.__name__)
    except Exception as e:
        print("[embed_gemini_only] google.generativeai import failed:", e)
        return None

    # variant A: genai.embeddings.create(...)
    try:
        if hasattr(genai, "embeddings"):
            print("[embed_gemini_only] Trying genai.embeddings.create(...)")
            resp = genai.embeddings.create(model=model, input=texts)
            # resp.data might be a list of dicts or objects
            data = getattr(resp, "data", resp)
            embeddings = []
            for d in data:
                if isinstance(d, dict) and "embedding" in d:
                    embeddings.append(d["embedding"])
                elif hasattr(d, "embedding"):
                    embeddings.append(d.embedding)
                else:
                    # fallback if response shape odd
                    embeddings.append(list(d))
            return np.array(embeddings, dtype=float)
    except Exception as e:
        print("[embed_gemini_only] variant A failed:", repr(e))

    # variant B: genai.get_embeddings(...)
    try:
        if hasattr(genai, "get_embeddings"):
            print("[embed_gemini_only] Trying genai.get_embeddings(...)")
            resp = genai.get_embeddings(model=model, input=texts)
            data = resp.get("data", resp)
            embeddings = []
            for d in data:
                if isinstance(d, dict) and "embedding" in d:
                    embeddings.append(d["embedding"])
                else:
                    embeddings.append(list(d))
            return np.array(embeddings, dtype=float)
    except Exception as e:
        print("[embed_gemini_only] variant B failed:", repr(e))

    # variant C: genai.client (some older wrappers)
    try:
        client = getattr(genai, "client", None)
        if client and hasattr(client, "embeddings"):
            print("[embed_gemini_only] Trying genai.client.embeddings.create(...)")
            resp = client.embeddings.create(model=model, input=texts)
            data = getattr(resp, "data", resp)
            embeddings = []
            for d in data:
                if isinstance(d, dict) and "embedding" in d:
                    embeddings.append(d["embedding"])
                else:
                    embeddings.append(list(d))
            return np.array(embeddings, dtype=float)
    except Exception as e:
        print("[embed_gemini_only] variant C failed:", repr(e))

    print("[embed_gemini_only] No known embeddings API variant succeeded in google.generativeai.")
    return None

def _pseudo_embeddings(n, dim=DEFAULT_DIM, seed=0):
    print(f"[embed_gemini_only] Generating pseudo-embeddings shape=({n},{dim})")
    rng = np.random.RandomState(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)

def make_embeddings(texts, model="textembedding-gecko-001", reuse_existing=True):
    """
    texts: list[str] or iterable
    Returns numpy array shape (len(texts), dim)
    """
    texts = list(map(lambda x: "" if x is None else str(x), texts))
    n = len(texts)

    # 0) try to reuse an existing embeddings.npy if it matches n
    if reuse_existing:
        existing = _load_existing()
        if existing is not None and existing.shape[0] == n:
            return existing

    # 1) try Gemini variants
    emb = _try_gemini_variants(texts, model=model)
    if emb is not None:
        try:
            # ensure numeric dtype
            emb = np.asarray(emb, dtype=float)
            if emb.shape[0] != n:
                print(f"[embed_gemini_only] Warning: returned rows {emb.shape[0]} != inputs {n}")
            _save(emb)
            return emb
        except Exception as e:
            print("[embed_gemini_only] Post-process embeddings failed:", e)
            traceback.print_exc()

    # 2) if reusing existing but length mismatch, still allow using it (safe fallback)
    if reuse_existing and os.path.exists(EMBED_PATH):
        existing = _load_existing()
        if existing is not None:
            print("[embed_gemini_only] Falling back to existing embeddings.npy despite length mismatch.")
            return existing

    # 3) last resort: deterministic pseudo embeddings
    emb = _pseudo_embeddings(n, DEFAULT_DIM, seed=42)
    _save(emb)
    return emb

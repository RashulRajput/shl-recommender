import joblib, numpy as np, sys, os
try:
    df = joblib.load("models/assessments_df.pkl")
    emb = np.load("models/embeddings.npy")
    tfm = joblib.load("models/tfidf_matrix.pkl")
    print("Loaded OK")
    print("df rows:", len(df))
    print("embeddings shape:", getattr(emb,'shape',None))
    print("tfidf_matrix shape:", getattr(tfm,'shape',None))
    if not (len(df) == emb.shape[0] == tfm.shape[0]):
        raise SystemError("MISMATCH: df / embeddings / tfidf sizes are different")
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)

import joblib, numpy as np, sys
try:
    df = joblib.load("models/assessments_df.pkl")
    emb = np.load("models/embeddings.npy")
    tf = joblib.load("models/tfidf.pkl")
    tfm = joblib.load("models/tfidf_matrix.pkl")
    print("Loaded OK")
    print("df shape:", getattr(df,'shape',None))
    print("embeddings shape:", getattr(emb,'shape',None))
    print("tfidf_matrix shape:", getattr(tfm,'shape',None))
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)

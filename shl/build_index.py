"""Build vector indexes for SHL assessments using Gemini embeddings."""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

os.makedirs("models", exist_ok=True)

def join_text(row: pd.Series) -> str:
	return " ".join(str(row.get(col, "")) for col in ["title", "raw_text", "test_type"])


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
	"""Create embeddings for a list of texts using Gemini.

	Tries the modern API (genai.embeddings.create). If unavailable, falls back to
	genai.embed_content for older library versions.
	"""
	embeddings = []
	for text in texts:
		emb = None
		# Try new embeddings API first
		try:
			if hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
				resp = genai.embeddings.create(model=model_name, input=text)
				emb = resp.data[0].embedding
		except Exception:
			emb = None
		# Fallback to legacy API
		if emb is None:
			fallback_model = os.getenv("GEMINI_EMBED_MODEL_FALLBACK", "models/embedding-001")
			result = genai.embed_content(model=fallback_model, content=text, task_type="retrieval_document")
			emb = result["embedding"]
		embeddings.append(emb)
	return np.array(embeddings, dtype=np.float32)

def build():
	df = pd.read_csv("data/assessments.csv")
	if df.empty:
		raise RuntimeError("data/assessments.csv is empty; run crawler first")
	
	# Filter to only rows with test_type (actual assessments, not navigation pages)
	df = df[df["test_type"].notnull()].copy()
	print(f"Loaded {len(df)} assessments with test_type")
	
	if df.empty:
		raise RuntimeError("No valid assessments found with test_type; check crawler output")
	
	df["text"] = df.apply(join_text, axis=1)

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY must be set to build embeddings")
	genai.configure(api_key=api_key)
	# Use the same default model as app.py for consistency
	model_name = os.getenv("GEMINI_EMBED_MODEL", "textembedding-gecko-001")

	# TF-IDF - with simpler parameters to avoid empty vocabulary
	print(f"Building TF-IDF for {len(df)} documents...")
	print(f"Sample text length: {len(df['text'].iloc[0]) if len(df) > 0 else 0}")
	tf = TfidfVectorizer(min_df=1, token_pattern=r'(?u)\b\w+\b')
	tfidf_matrix = tf.fit_transform(df["text"].values)
	print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
	joblib.dump(tf, "models/tfidf.pkl")
	joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")

	texts = df["text"].tolist()
	# Try to (re)build embeddings; if the API quota is exceeded, keep existing file.
	try:
		print(f"Building embeddings with model '{model_name}'...")
		embeddings_chunks = []
		batch_size = max(1, int(os.getenv("GEMINI_EMBED_BATCH", "10")))
		for start in range(0, len(texts), batch_size):
			batch = texts[start:start + batch_size]
			embeddings_chunks.append(embed_texts(batch, model_name))
		embeddings = np.vstack(embeddings_chunks)
		np.save("models/embeddings.npy", embeddings)
		print(f"Embeddings built: shape {embeddings.shape}")
	except Exception as e:
		print(f"[build] Skipping embeddings generation due to error: {e}")
		if os.path.exists("models/embeddings.npy"):
			print("Using existing models/embeddings.npy")
		else:
			print("No existing embeddings.npy found; the app will fall back to TF-IDF only at runtime.")

	joblib.dump(df, "models/assessments_df.pkl")
	print("Index built successfully.")

if __name__ == "__main__":
	build()

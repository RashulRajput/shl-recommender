"""Build TF-IDF + SentenceTransformer index locally.

This script is intended for local preprocessing. It uses sentence-transformers
to generate embeddings that are committed or deployed alongside the service.
Runtime query embeddings in the API may use Gemini; catalog embeddings here
remain ST-based to avoid quota consumption.
"""

import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

os.makedirs("models", exist_ok=True)

def join_text(row: pd.Series) -> str:
	return " ".join(str(row.get(col, "")) for col in ["title", "raw_text", "test_type"])


def embed_texts_st(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
	return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def build():
	df = pd.read_csv("data/assessments.csv")
	if df.empty:
		raise RuntimeError("data/assessments.csv is empty; run crawler first")
	df["text"] = df.apply(join_text, axis=1)

	# SentenceTransformer model name (can be parameterized)
	st_model_name = os.getenv("ST_MODEL_NAME", "all-MiniLM-L6-v2")
	st_model = SentenceTransformer(st_model_name)

	# TF-IDF
	tf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
	tfidf_matrix = tf.fit_transform(df["text"].values)
	joblib.dump(tf, "models/tfidf.pkl")
	joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")

	# Embeddings
	texts = df["text"].tolist()
	embeddings = embed_texts_st(st_model, texts)
	np.save("models/embeddings.npy", embeddings)

	joblib.dump(df, "models/assessments_df.pkl")
	joblib.dump(st_model, "models/embedder.pkl")
	print("Index built successfully (SentenceTransformer)")

if __name__ == "__main__":
	build()

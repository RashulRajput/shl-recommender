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
	embeddings = []
	for text in texts:
		result = genai.embed_content(model=model_name, content=text, task_type="retrieval_document")
		embeddings.append(result['embedding'])
	return np.array(embeddings, dtype=np.float32)

def build():
	df = pd.read_csv("data/assessments.csv")
	if df.empty:
		raise RuntimeError("data/assessments.csv is empty; run crawler first")
	df["text"] = df.apply(join_text, axis=1)

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY must be set to build embeddings")
	genai.configure(api_key=api_key)
	model_name = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")

	# TF-IDF
	tf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
	tfidf_matrix = tf.fit_transform(df["text"].values)
	joblib.dump(tf, "models/tfidf.pkl")
	joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")

	texts = df["text"].tolist()
	embeddings_chunks = []
	batch_size = max(1, int(os.getenv("GEMINI_EMBED_BATCH", "10")))
	for start in range(0, len(texts), batch_size):
		batch = texts[start:start + batch_size]
		embeddings_chunks.append(embed_texts(batch, model_name))
	embeddings = np.vstack(embeddings_chunks)
	np.save("models/embeddings.npy", embeddings)

	joblib.dump(df, "models/assessments_df.pkl")
	print("Index built successfully.")

if __name__ == "__main__":
	build()

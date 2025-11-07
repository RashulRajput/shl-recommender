import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.makedirs("models", exist_ok=True)

def join_text(row):
	return " ".join(str(row.get(c, "")) for c in ["title", "raw_text", "test_type"])

def build():
	df = pd.read_csv("data/assessments.csv")
	df["text"] = df.apply(join_text, axis=1)

	# TF-IDF
	tf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
	tfidf_matrix = tf.fit_transform(df["text"].values)
	joblib.dump(tf, "models/tfidf.pkl")
	joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")

	# Sentence embeddings
	model = SentenceTransformer("all-MiniLM-L6-v2")
	embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
	np.save("models/embeddings.npy", embeddings)

	joblib.dump(df, "models/assessments_df.pkl")
	joblib.dump(model, "models/embedder.pkl")
	print("Index built successfully.")

if __name__ == "__main__":
	build()

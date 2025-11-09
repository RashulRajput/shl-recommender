"""Build vector indexes for SHL assessments using Gemini embeddings."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai
from embed_gemini_only import make_embeddings

os.makedirs("models", exist_ok=True)

# if Gen_AI train/test files exist, load and append mappings if necessary
def merge_mapping(path):
	if os.path.exists(path):
		mdf = pd.read_csv(path)
		# expect columns Query, Assessment_url or similar - normalize
		if 'Assessment_url' in mdf.columns or 'assessment_url' in mdf.columns:
			# optionally use for downstream eval only
			print(f"Loaded mapping {path} ({len(mdf)})")
		else:
			print(f"Loaded unlabelled mapping {path}")

def join_text(row: pd.Series) -> str:
	return " ".join(str(row.get(col, "")) for col in ["title", "raw_text", "test_type"])

def build():
	"""Build or refresh vector artifacts.

	Priority for consistency:
	- If models/assessments_df.pkl exists, use it as the source of truth so that
	  TF-IDF and embeddings align in row count and order.
	- Otherwise, construct from data/assessments.csv (filtering out rows without test_type),
	  and attempt to build embeddings as well.
	"""

	model_df_path = "models/assessments_df.pkl"
	if os.path.exists(model_df_path):
		print("Loading existing models/assessments_df.pkl to preserve row alignment...")
		df = joblib.load(model_df_path)
		# Ensure required columns exist; reconstruct text if needed
		if "text" not in df.columns or df["text"].isna().all():
			print("Reconstructing 'text' column from title/raw_text/test_type...")
			df["text"] = df.apply(join_text, axis=1)
	else:
		# Build from raw catalog
		df = pd.read_csv("data/assessments.csv")
		if df.empty:
			raise RuntimeError("data/assessments.csv is empty; run crawler first")
		# keep all products — fill missing test_type with 'general'
		df['test_type'] = df['test_type'].fillna('general')
		print(f"Loaded {len(df)} assessments with test_type from catalog")
		if df.empty:
			raise RuntimeError("No valid assessments found with test_type; check crawler output")
		df["text"] = df.apply(join_text, axis=1)
		
		# Load train/test mappings if they exist
		train_map_path = os.path.join("data", "Gen_AI_Train-Set_FULL.csv")
		test_map_path = os.path.join("data", "Gen_AI_Test-Set_FULL.csv")
		merge_mapping(train_map_path)
		merge_mapping(test_map_path)

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

	texts = df["text"].astype(str).tolist()
	# Use the new embed_gemini_only wrapper which handles fallbacks gracefully
	print(f"Building embeddings with model '{model_name}' using embed_gemini_only wrapper...")
	embeddings = make_embeddings(texts, model=model_name, reuse_existing=True)
	print(f"Embeddings ready: shape {embeddings.shape}")

	joblib.dump(df, "models/assessments_df.pkl")
	print("Index built successfully.")

if __name__ == "__main__":
	build()

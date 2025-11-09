"""
Quick test script for the Java developer query
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shl'))

# Load models first
os.chdir('shl')
import app
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Loading models...")
app.load_models()

if not app.MODELS_READY:
    print("ERROR: Models not loaded!")
    sys.exit(1)

print(f"✓ Models loaded: {len(app.df)} assessments")
print(f"✓ Embeddings shape: {app.embeddings.shape if app.embeddings is not None else 'None'}")
print(f"✓ TF-IDF matrix shape: {app.tfidf_matrix.shape}")
print()

# Test the original query
query = "I am hiring for Java developers who can also collaborate effectively with my business teams"
print(f"Query: {query}\n")
print("=" * 80)

try:
    # Get TF-IDF vector
    q_vec = app.tf.transform([query])
    
    # Get query embedding if available
    if app.embeddings is not None:
        from embed_gemini_only import make_embeddings
        q_emb = make_embeddings([query], reuse_existing=False)
        if q_emb is not None and q_emb.ndim == 2:
            q_emb = q_emb.ravel()  # Flatten to 1D
    else:
        q_emb = None
    
    # Use optimized hybrid_score function
    if q_emb is not None:
        final = app.hybrid_score(q_vec, app.tfidf_matrix, q_emb, app.embeddings, alpha=0.6)
    else:
        # Fallback to TF-IDF only
        tfidf_sims = cosine_similarity(q_vec, app.tfidf_matrix).flatten()
        final = app.normalize(tfidf_sims)
    
    # Get top 5
    top_idx = np.argsort(final)[::-1][:5]
    
    print(f"\nTop 5 Recommendations:\n")
    for rank, idx in enumerate(top_idx, 1):
        row = app.df.iloc[idx]
        print(f"{rank}. {row['title']}")
        print(f"   Score: {final[idx]:.4f}")
        print(f"   Type: {row['test_type']}")
        print(f"   URL: {row['url'][:80]}...")
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

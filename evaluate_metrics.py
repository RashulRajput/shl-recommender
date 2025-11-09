"""
Evaluate recommendation system using Mean Recall@K metric
"""
import pandas as pd
import requests
import sys
from collections import defaultdict
import re

API_URL = "http://127.0.0.1:8002/recommend"
TRAINING_DATA = "shl/data/Gen_AI_Train-Set_FULL.csv"

def normalize_url(url):
    """
    Normalize SHL URLs to handle format differences:
    - Remove /solutions/ from path
    - Remove trailing slashes
    - Extract assessment ID for comparison
    """
    url = str(url).strip()
    # Remove /solutions/ from path
    url = url.replace('/solutions/products/', '/products/')
    # Remove trailing slash
    url = url.rstrip('/')
    return url

def get_assessment_id(url):
    """Extract assessment ID from URL for fuzzy matching"""
    # Extract the last part of the URL (assessment ID)
    match = re.search(r'/view/([^/]+)/?$', url)
    if match:
        return match.group(1)
    return url

def load_training_data():
    """Load training data with ground truth labels"""
    df = pd.read_csv(TRAINING_DATA)
    
    # Group by query to get all relevant assessments per query
    query_assessments = defaultdict(set)
    for _, row in df.iterrows():
        query = str(row['Query']).strip()
        assessment_url = str(row['Assessment_url']).strip()
        # Normalize URLs
        normalized_url = normalize_url(assessment_url)
        query_assessments[query].add(normalized_url)
    
    return dict(query_assessments)

def get_recommendations(query, top_k=5):
    """Get top K recommendations from API"""
    try:
        payload = {"job_title": query, "description": "", "top_k": top_k}
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract URLs from recommendations and normalize them
        recommended_urls = [normalize_url(rec['assessment_url']) for rec in result.get('recommendations', [])]
        return recommended_urls
    except Exception as e:
        print(f"Error getting recommendations for '{query[:50]}...': {e}")
        return []

def calculate_recall_at_k(relevant_urls, recommended_urls):
    """
    Calculate Recall@K for a single query
    Recall@K = (Number of relevant items in top K) / (Total relevant items)
    """
    if not relevant_urls:
        return 0.0
    
    # Count how many recommended URLs are in the relevant set
    relevant_in_topk = len(set(recommended_urls) & set(relevant_urls))
    total_relevant = len(relevant_urls)
    
    return relevant_in_topk / total_relevant

def evaluate_mean_recall_at_k(k_values=[5, 10]):
    """
    Evaluate Mean Recall@K across all training queries
    """
    print("Loading training data...")
    query_to_assessments = load_training_data()
    
    print(f"Found {len(query_to_assessments)} unique queries in training data")
    print(f"Evaluating with K values: {k_values}\n")
    print("=" * 80)
    
    results = {}
    
    for K in k_values:
        print(f"\n📊 Evaluating Mean Recall@{K}...")
        print("-" * 80)
        
        recall_scores = []
        
        for query, relevant_urls in query_to_assessments.items():
            # Get recommendations
            recommended_urls = get_recommendations(query, top_k=K)
            
            # Calculate recall for this query
            recall = calculate_recall_at_k(relevant_urls, recommended_urls)
            recall_scores.append(recall)
            
            # Print details
            print(f"\nQuery: {query[:70]}...")
            print(f"  Relevant assessments: {len(relevant_urls)}")
            print(f"  Retrieved in top-{K}: {len(set(recommended_urls) & set(relevant_urls))}")
            print(f"  Recall@{K}: {recall:.4f}")
        
        # Calculate mean recall
        mean_recall = sum(recall_scores) / len(recall_scores)
        results[K] = {
            'mean_recall': mean_recall,
            'scores': recall_scores,
            'num_queries': len(recall_scores)
        }
        
        print("\n" + "=" * 80)
        print(f"📈 Mean Recall@{K}: {mean_recall:.4f}")
        print(f"   Min: {min(recall_scores):.4f}")
        print(f"   Max: {max(recall_scores):.4f}")
        print(f"   Queries evaluated: {len(recall_scores)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL RESULTS")
    print("=" * 80)
    for K, metrics in results.items():
        print(f"Mean Recall@{K}: {metrics['mean_recall']:.4f}")
    
    return results

def main():
    try:
        # Test API connection first
        print("Testing API connection...")
        test_response = requests.get("http://127.0.0.1:8002/health", timeout=5)
        if test_response.status_code != 200:
            print("❌ API is not running! Start the server first:")
            print("   cd shl")
            print("   python -m uvicorn app:app --host 127.0.0.1 --port 8002")
            sys.exit(1)
        print("✓ API is running\n")
        
        # Run evaluation
        results = evaluate_mean_recall_at_k(k_values=[5, 10])
        
        # Save results
        output_file = "evaluation_results.txt"
        with open(output_file, 'w') as f:
            f.write("SHL Recommendation System - Evaluation Results\n")
            f.write("=" * 80 + "\n\n")
            for K, metrics in results.items():
                f.write(f"Mean Recall@{K}: {metrics['mean_recall']:.4f}\n")
        
        print(f"\n✓ Results saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Training data file not found: {TRAINING_DATA}")
        print("   Make sure you're running from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

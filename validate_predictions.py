"""
Validate and analyze predictions for the test set
"""
import pandas as pd
import ast

def parse_list(x):
    """Parse list from string representation"""
    try:
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            # Try parsing as Python literal
            return ast.literal_eval(x)
        else:
            return []
    except:
        return []

def main():
    print("=" * 80)
    print("SHL Test Set Predictions - Validation & Analysis")
    print("=" * 80)
    
    # ---- Load files ----
    print("\n📂 Loading files...")
    try:
        test_df = pd.read_csv("shl/data/Gen_AI_Test-Set_FULL.csv")
        pred_df = pd.read_csv("shl/predictions.csv")
        print(f"✓ Test set loaded: {len(test_df)} queries")
        print(f"✓ Predictions loaded: {len(pred_df)} rows")
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        return
    
    # ---- Basic Validation ----
    print("\n" + "=" * 80)
    print("📋 BASIC VALIDATION")
    print("=" * 80)
    
    # Check required columns
    required_cols = ['Query', 'Assessment_url']
    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        print(f"❌ Missing columns in predictions: {missing_cols}")
        return
    else:
        print(f"✓ All required columns present: {required_cols}")
    
    # Check number of predictions per query
    print("\n📊 Predictions per query:")
    pred_counts = pred_df.groupby('Query').size()
    print(f"   Total queries in predictions: {len(pred_counts)}")
    print(f"   Total queries in test set: {len(test_df)}")
    print(f"   Min predictions/query: {pred_counts.min()}")
    print(f"   Max predictions/query: {pred_counts.max()}")
    print(f"   Average predictions/query: {pred_counts.mean():.2f}")
    
    if len(pred_counts) != len(test_df):
        print(f"⚠️  Warning: Prediction count ({len(pred_counts)}) doesn't match test set ({len(test_df)})")
        missing_queries = set(test_df['Query']) - set(pred_df['Query'])
        if missing_queries:
            print(f"   Missing queries: {len(missing_queries)}")
            for q in list(missing_queries)[:3]:
                print(f"     - {q[:70]}...")
    else:
        print("✓ All test queries have predictions")
    
    # Check for duplicate recommendations within same query
    print("\n🔍 Checking for duplicates within queries...")
    duplicates_found = False
    for query, group in pred_df.groupby('Query'):
        url_counts = group['Assessment_url'].value_counts()
        if (url_counts > 1).any():
            duplicates_found = True
            print(f"⚠️  Query has duplicate URLs:")
            print(f"   Query: {query[:70]}...")
            for url, count in url_counts[url_counts > 1].items():
                print(f"   - {url} appears {count} times")
    
    if not duplicates_found:
        print("✓ No duplicate URLs within queries")
    
    # Check URL format
    print("\n🔗 Validating URL format...")
    invalid_urls = pred_df[~pred_df['Assessment_url'].astype(str).str.startswith('https://www.shl.com/')]
    if len(invalid_urls) > 0:
        print(f"⚠️  Found {len(invalid_urls)} potentially invalid URLs")
        print("   First few:")
        for url in invalid_urls['Assessment_url'].head(3):
            print(f"     - {url}")
    else:
        print("✓ All URLs have correct SHL format")
    
    # ---- Quality Analysis ----
    print("\n" + "=" * 80)
    print("📈 PREDICTION QUALITY ANALYSIS")
    print("=" * 80)
    
    # Score distribution (if Score column exists)
    if 'Score' in pred_df.columns:
        print("\n📊 Score distribution:")
        print(f"   Mean score: {pred_df['Score'].mean():.4f}")
        print(f"   Median score: {pred_df['Score'].median():.4f}")
        print(f"   Min score: {pred_df['Score'].min():.4f}")
        print(f"   Max score: {pred_df['Score'].max():.4f}")
        print(f"   Std dev: {pred_df['Score'].std():.4f}")
        
        # Top scores per query
        print("\n🏆 Top recommendation score per query:")
        for query in pred_df['Query'].unique():
            query_preds = pred_df[pred_df['Query'] == query]
            top_score = query_preds['Score'].max()
            top_url = query_preds[query_preds['Score'] == top_score]['Assessment_url'].iloc[0]
            print(f"\n   Query: {query[:70]}...")
            print(f"   Top score: {top_score:.4f}")
            print(f"   Top assessment: {top_url.split('/')[-2] if '/' in top_url else top_url}")
    
    # Assessment diversity
    print("\n" + "=" * 80)
    print("🎯 ASSESSMENT DIVERSITY")
    print("=" * 80)
    unique_assessments = pred_df['Assessment_url'].nunique()
    total_predictions = len(pred_df)
    print(f"   Unique assessments recommended: {unique_assessments}")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Diversity ratio: {unique_assessments/total_predictions:.2%}")
    
    # Most frequently recommended
    print("\n📌 Most frequently recommended assessments:")
    top_assessments = pred_df['Assessment_url'].value_counts().head(10)
    for i, (url, count) in enumerate(top_assessments.items(), 1):
        assessment_name = url.split('/')[-2] if '/' in url else url
        print(f"   {i}. {assessment_name}: {count} times")
    
    # ---- Per-Query Summary ----
    print("\n" + "=" * 80)
    print("📝 PER-QUERY SUMMARY")
    print("=" * 80)
    
    for i, query in enumerate(test_df['Query'].unique(), 1):
        query_preds = pred_df[pred_df['Query'] == query]
        print(f"\n{i}. Query: {query[:70]}...")
        print(f"   Predictions: {len(query_preds)}")
        if 'Score' in pred_df.columns and len(query_preds) > 0:
            print(f"   Score range: {query_preds['Score'].min():.4f} - {query_preds['Score'].max():.4f}")
            print(f"   Top 3 recommendations:")
            for j, (_, row) in enumerate(query_preds.nlargest(3, 'Score').iterrows(), 1):
                assessment_name = row['Assessment_url'].split('/')[-2] if '/' in row['Assessment_url'] else row['Assessment_url']
                print(f"      {j}) {assessment_name} (score: {row['Score']:.4f})")
    
    # ---- Export Summary ----
    print("\n" + "=" * 80)
    print("💾 EXPORT VALIDATION")
    print("=" * 80)
    
    # Check if submission format exists
    try:
        pred_min = pd.read_csv("shl/predictions_min.csv")
        print(f"✓ Submission file exists: shl/predictions_min.csv")
        print(f"   Rows: {len(pred_min)}")
        print(f"   Columns: {list(pred_min.columns)}")
        
        # Validate submission format
        if set(pred_min.columns) == set(['Query', 'Assessment_url']):
            print("✓ Submission format correct (Query, Assessment_url)")
        else:
            print(f"⚠️  Unexpected columns in submission file")
    except FileNotFoundError:
        print("⚠️  Submission file not found: shl/predictions_min.csv")
        print("   Create it by removing Score column from predictions.csv")
    
    # ---- Final Summary ----
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)
    print(f"✓ {len(pred_counts)} test queries processed")
    print(f"✓ {len(pred_df)} total predictions generated")
    print(f"✓ {unique_assessments} unique assessments recommended")
    if 'Score' in pred_df.columns:
        print(f"✓ Average confidence score: {pred_df['Score'].mean():.4f}")
    
    print("\n📁 Files ready for submission:")
    print("   - shl/predictions.csv (with scores)")
    print("   - shl/predictions_min.csv (submission format)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

"""
Validation script for predictions_min.csv submission file.
Ensures format matches grader expectations.
"""
import pandas as pd
import sys
import os

def validate_submission():
    file_path = 'predictions_min.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: {file_path} not found")
        sys.exit(1)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ ERROR: Cannot read CSV: {e}")
        sys.exit(1)
    
    # Check headers
    expected_cols = ['Query', 'Assessment_url']
    if list(df.columns) != expected_cols:
        print(f"❌ ERROR: Wrong headers. Expected {expected_cols}, got {list(df.columns)}")
        sys.exit(1)
    
    # Check row count
    if df.shape[0] == 0:
        print("❌ ERROR: No rows in file")
        sys.exit(1)
    
    # Check for nulls
    if df.isnull().any().any():
        print("⚠️  WARNING: File contains null values")
        print(df.isnull().sum())
    
    # Check URL format
    if not df['Assessment_url'].str.startswith('http').all():
        print("⚠️  WARNING: Some Assessment_url values don't start with http")
    
    print(f"✅ predictions_min.csv validation passed!")
    print(f"   Rows: {df.shape[0]}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample:")
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    validate_submission()

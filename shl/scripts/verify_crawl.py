#!/usr/bin/env python3
"""
Verification script for assessments_full.csv
Checks integrity, coverage, and quality of the crawled SHL catalog.
"""

import csv
import os
import sys
from collections import Counter

def verify_assessments(csv_path):
    """Validate the assessments CSV file."""
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        return False
    
    print(f"📄 Verifying: {csv_path}")
    print("=" * 70)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"❌ ERROR: Failed to read CSV: {e}")
        return False
    
    # Basic stats
    total = len(rows)
    print(f"\n✅ Total assessments: {total}")
    
    if total == 0:
        print("❌ ERROR: Empty CSV file")
        return False
    
    # Check required columns
    required_cols = ['title', 'url', 'test_type', 'raw_text', 'text']
    actual_cols = set(rows[0].keys())
    missing_cols = set(required_cols) - actual_cols
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {missing_cols}")
        return False
    else:
        print(f"✅ All required columns present: {required_cols}")
    
    # Check for empty/missing data
    empty_titles = sum(1 for r in rows if not r.get('title', '').strip())
    empty_urls = sum(1 for r in rows if not r.get('url', '').strip())
    empty_types = sum(1 for r in rows if not r.get('test_type', '').strip())
    
    print(f"\n📊 Data Quality:")
    print(f"   - Empty titles: {empty_titles}")
    print(f"   - Empty URLs: {empty_urls}")
    print(f"   - Empty test_types: {empty_types}")
    
    if empty_titles > total * 0.1:  # More than 10% empty
        print(f"   ⚠️  WARNING: {empty_titles} assessments have empty titles")
    
    # Check test_type distribution
    test_types = Counter(r.get('test_type', '').strip() for r in rows if r.get('test_type', '').strip())
    print(f"\n📈 Test Type Distribution:")
    for tt, count in test_types.most_common():
        pct = (count / total) * 100
        print(f"   - {tt:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Check for duplicates
    urls = [r.get('url', '') for r in rows]
    unique_urls = set(urls)
    duplicates = len(urls) - len(unique_urls)
    
    if duplicates > 0:
        print(f"\n⚠️  WARNING: {duplicates} duplicate URLs found")
    else:
        print(f"\n✅ No duplicate URLs")
    
    # Check for key assessments (training data requirements)
    key_assessments = [
        'Core Java (Entry Level)',
        'Core Java (Advanced Level)',
        'Python',
        'SQL',
        'Selenium',
        'Automata',
        'Interpersonal Communications',
        'JavaScript',
        'HTML',
        'CSS',
    ]
    
    titles_lower = [r.get('title', '').lower() for r in rows]
    found_key = []
    missing_key = []
    
    for key in key_assessments:
        found = any(key.lower() in title for title in titles_lower)
        if found:
            found_key.append(key)
        else:
            missing_key.append(key)
    
    print(f"\n🎯 Key Assessments (Training Data Requirements):")
    print(f"   Found: {len(found_key)}/{len(key_assessments)}")
    if found_key:
        for k in found_key:
            print(f"   ✅ {k}")
    if missing_key:
        print(f"\n   ⚠️  Missing:")
        for k in missing_key:
            print(f"   ❌ {k}")
    
    # URL validation
    invalid_urls = sum(1 for r in rows if not r.get('url', '').startswith('http'))
    if invalid_urls > 0:
        print(f"\n⚠️  WARNING: {invalid_urls} assessments have invalid URLs")
    
    # Text content check
    short_texts = sum(1 for r in rows if len(r.get('raw_text', '')) < 100)
    print(f"\n📝 Content Quality:")
    print(f"   - Assessments with short descriptions (<100 chars): {short_texts}")
    if short_texts > total * 0.2:  # More than 20%
        print(f"   ⚠️  WARNING: Many assessments have minimal content")
    
    # Final summary
    print("\n" + "=" * 70)
    
    issues = []
    if empty_titles > 0:
        issues.append(f"{empty_titles} empty titles")
    if empty_urls > 0:
        issues.append(f"{empty_urls} empty URLs")
    if duplicates > 0:
        issues.append(f"{duplicates} duplicates")
    if missing_key:
        issues.append(f"{len(missing_key)} missing key assessments")
    
    if not issues:
        print("✅ VERIFICATION PASSED - All checks OK!")
        print(f"   Ready to use {total} assessments for recommendation system")
        return True
    else:
        print("⚠️  VERIFICATION COMPLETED WITH WARNINGS:")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\n   Total assessments: {total}")
        print(f"   You may proceed, but consider reviewing the warnings above.")
        return True  # Still usable despite warnings


if __name__ == "__main__":
    # Default path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(script_dir, "..", "data", "assessments_full.csv")
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    success = verify_assessments(csv_path)
    sys.exit(0 if success else 1)

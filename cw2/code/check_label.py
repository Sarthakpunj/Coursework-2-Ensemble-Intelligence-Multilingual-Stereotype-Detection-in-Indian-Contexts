"""
Quick Diagnostic: Check CSV Label Format
Run this first to see what labels your CSV actually has
"""

import pandas as pd
from pathlib import Path

# Update this path to match your CSV location
CSV_PATH = Path('/Users/sarthakpunj/Downloads/bias_detection_COMPLETE/test_ULTIMATE.csv')

print("="*60)
print("CHECKING CSV LABELS")
print("="*60)

# Load CSV
df = pd.read_csv(CSV_PATH)

print(f"\n📊 Total rows: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")

# Check label column
print("\n" + "="*60)
print("LABEL ANALYSIS")
print("="*60)

print(f"\nLabel column name: 'label'")
print(f"Label data type: {df['label'].dtype}")
print(f"\nUnique labels: {df['label'].unique()}")
print(f"\nLabel counts:")
print(df['label'].value_counts())

# Check a few sample rows
print("\n" + "="*60)
print("SAMPLE ROWS")
print("="*60)
print(df[['text', 'label']].head(10))

# Determine correct mapping
print("\n" + "="*60)
print("RECOMMENDED MAPPING")
print("="*60)

if df['label'].dtype == 'object':
    unique_labels = df['label'].unique()
    print("\n✅ Labels are STRINGS")
    print(f"Labels found: {unique_labels}")
    
    if 'non-stereotype' in unique_labels or 'non_stereotype' in unique_labels:
        print("\n📌 Suggested mapping:")
        print("   'non-stereotype' or 'non_stereotype' → 0")
        print("   'stereotype' → 1")
    elif 'Non-Stereotype' in unique_labels:
        print("\n📌 Suggested mapping:")
        print("   'Non-Stereotype' → 0")
        print("   'Stereotype' → 1")
    else:
        print(f"\n⚠️  Unknown labels: {unique_labels}")
        print("Please verify the correct mapping!")
else:
    print("\n✅ Labels are already NUMERIC")
    print(f"Label values: {df['label'].unique()}")
    print("\nNo mapping needed - labels are already 0/1")

print("\n" + "="*60)
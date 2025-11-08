# shl/scripts/fix_assessments.py
# Usage: venv\Scripts\python.exe shl\scripts\fix_assessments.py
import os, sys
import pandas as pd
import csv
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo/shl
DATA_DIR = os.path.join(ROOT, "data")
EXCEL_PATH = os.path.join(DATA_DIR, "Gen_AI Dataset.xlsx")
ASSESS_CSV = os.path.join(DATA_DIR, "assessments.csv")
BACKUP = os.path.join(DATA_DIR, "assessments.csv.bak")

def is_numeric_title(s):
    if pd.isna(s): return True
    s = str(s).strip()
    return s == "" or re.fullmatch(r"\d+", s) is not None

def normalize_cols(df):
    # prefer lowercased columns for guesses
    header = {c.lower(): c for c in df.columns}
    rename_map = {}
    if "title" not in header:
        # try common names
        for guess in ("assessment_name","assessment","name"):
            if guess in header:
                rename_map[header[guess]] = "title"
                break
    if "url" not in header:
        for guess in ("assessment_url","url","link"):
            if guess in header:
                rename_map[header[guess]] = "url"
                break
    if "text" not in header:
        for guess in ("text","description","query","query_text","content"):
            if guess in header:
                rename_map[header[guess]] = "text"
                break
    df = df.rename(columns=rename_map)
    # ensure columns exist
    if "title" not in df.columns: df["title"] = ""
    if "url" not in df.columns: df["url"] = ""
    if "text" not in df.columns: df["text"] = df["title"].astype(str)
    return df[["title","url","text"]]

def main():
    if not os.path.exists(EXCEL_PATH):
        print("ERROR: Excel file not found at", EXCEL_PATH)
        sys.exit(2)

    # load all sheets and concat (handles odd excel layouts)
    xls = pd.read_excel(EXCEL_PATH, sheet_name=None)
    parts = []
    for name, df in xls.items():
        if df.shape[1] == 0: continue
        parts.append(df)
    if not parts:
        print("ERROR: No usable sheets in Excel.")
        sys.exit(2)
    df = pd.concat(parts, ignore_index=True, sort=False)
    df = normalize_cols(df)

    # Trim whitespace
    df["title"] = df["title"].astype(str).str.strip()
    df["url"] = df["url"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Keep rows that are not empty and not purely numeric titles
    mask_keep = ~df["title"].apply(is_numeric_title)
    cleaned = df[mask_keep].reset_index(drop=True)

    # If cleaned is empty, fallback: try using url pages or keep everything (but keep backup before)
    if cleaned.empty:
        print("WARNING: Cleaning removed all rows (no non-numeric titles).")
        print("I'll write a copy of the original read Excel as 'assessments_from_excel.csv' instead for inspection.")
        out_path = os.path.join(DATA_DIR, "assessments_from_excel.csv")
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print("Wrote", out_path)
        sys.exit(0)

    # Save backup of existing CSV (if exists)
    if os.path.exists(ASSESS_CSV):
        print("Backing up existing assessments.csv -> assessments.csv.bak")
        os.replace(ASSESS_CSV, BACKUP)

    # write cleaned CSV with all fields quoted
    cleaned.to_csv(ASSESS_CSV, index=False, quoting=csv.QUOTE_ALL)
    print("Wrote cleaned assessments.csv with", len(cleaned), "rows to", ASSESS_CSV)
    print("Backup (if existed) is at:", BACKUP)

if __name__ == "__main__":
    main()

"""
Crawl SHL product catalog and extract "Individual Test Solutions".
Outputs: data/assessments.csv
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import time
import re
import os
import argparse

BASE = "https://www.shl.com/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"}
os.makedirs("data", exist_ok=True)

def fetch(url, timeout=15):
	try:
		r = requests.get(url, headers=HEADERS, timeout=timeout)
		r.raise_for_status()
		return r.text
	except Exception as e:
		print(f"[fetch] {e}")
		return ""

def find_individual_section(soup):
	for htag in ("h1", "h2", "h3", "h4", "h5"):
		for h in soup.find_all(htag):
			txt = h.get_text(separator=" ", strip=True).lower()
			if "individual test" in txt:
				return h
	return None

def parse_catalog_page(html):
	soup = BeautifulSoup(html, "html.parser")
	heading = find_individual_section(soup)
	items = []
	if heading:
		for sibling in heading.find_all_next():
			if sibling.name in ("h1", "h2"):
				break
			for a in sibling.find_all("a", href=True):
				title = a.get_text(strip=True)
				href = urljoin(BASE, a["href"])
				if title and "/products/" in href:
					if "pre-packaged" not in title.lower():
						items.append((title, href))
	if not items:
		for a in soup.find_all("a", href=True):
			title = a.get_text(strip=True)
			href = urljoin(BASE, a["href"])
			if title and "/products/" in href and "pre-packaged" not in title.lower():
				items.append((title, href))
	seen, out = set(), []
	for t, h in items:
		if h not in seen:
			seen.add(h)
			out.append({"title": t, "url": h})
	return out

def extract_product_page(url):
    html = fetch(url)
    if not html:
        return {"test_type": None, "raw_text": "", "skip": False}
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    
    # Exclude pre-packaged solutions robustly
    if "pre-packaged" in text.lower() or "pre packaged" in text.lower():
        return {"test_type": None, "raw_text": "", "skip": True}
    
    m = re.search(r"Test\s*Type[:\s]*([A-Z, ]{1,6})", text, re.I)
    test_type = m.group(1).strip() if m else None
    return {"test_type": test_type, "raw_text": text, "skip": False}def crawl_catalog(max_pages=25, throttle=0.5):
	html = fetch(BASE)
	items = parse_catalog_page(html)
	for p in range(2, max_pages + 1):
		page_url = f"{BASE}?page={p}"
		h = fetch(page_url)
		if not h:
			break
		more = parse_catalog_page(h)
		if not more:
			break
		items.extend(more)
		time.sleep(throttle)
	df = pd.DataFrame(items).drop_duplicates(subset=["url"])
	print(f"Found {len(df)} product URLs")
	return df

def enrich_and_save(df, throttle=0.4):
    records = []
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] {row['title']}")
        meta = extract_product_page(row["url"])
        if meta.get("skip", False):
            print(f"  -> Skipping (pre-packaged)")
            continue
        records.append({
            "title": row["title"],
            "url": row["url"],
            "test_type": meta["test_type"],
            "raw_text": meta["raw_text"]
        })
        time.sleep(throttle)
    out = pd.DataFrame(records)
    out.to_csv("data/assessments.csv", index=False)
    print("Saved data/assessments.csv")
    return outif __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Crawl SHL catalog")
	parser.add_argument("--max-pages", type=int, default=25, help="Number of catalog pages to crawl")
	parser.add_argument("--throttle", type=float, default=0.5, help="Seconds to sleep between requests")
	args = parser.parse_args()

	df = crawl_catalog(max_pages=args.max_pages, throttle=args.throttle)
	if not df.empty:
		enrich_and_save(df, throttle=args.throttle)

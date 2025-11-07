"""
Crawl SHL product catalog and extract "Individual Test Solutions".
Outputs: data/assessments.csv
"""

import argparse
import os
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"}
os.makedirs("data", exist_ok=True)


def fetch(url, timeout=15):
	try:
		r = requests.get(url, headers=HEADERS, timeout=timeout)
		r.raise_for_status()
		return r.text
	except Exception as exc:  # network errors should not crash crawl
		print(f"[fetch] {exc}")
		return ""


def find_individual_section(soup):
	for htag in ("h1", "h2", "h3", "h4", "h5"):
		for heading in soup.find_all(htag):
			text = heading.get_text(separator=" ", strip=True).lower()
			if "individual test" in text:
				return heading
	return None


def parse_catalog_page(html):
	soup = BeautifulSoup(html, "html.parser")
	heading = find_individual_section(soup)
	items = []

	if heading:
		for sibling in heading.find_all_next():
			if sibling.name in ("h1", "h2"):
				break
			for anchor in sibling.find_all("a", href=True):
				title = anchor.get_text(strip=True)
				href = urljoin(BASE, anchor["href"])
				if title and "/products/" in href and "pre-packaged" not in title.lower():
					items.append((title, href))

	if not items:
		for anchor in soup.find_all("a", href=True):
			title = anchor.get_text(strip=True)
			href = urljoin(BASE, anchor["href"])
			if title and "/products/" in href and "pre-packaged" not in title.lower():
				items.append((title, href))

	unique, output = set(), []
	for title, href in items:
		if href not in unique:
			unique.add(href)
			output.append({"title": title, "url": href})
	return output


def extract_product_page(url):
	html = fetch(url)
	if not html:
		return {"test_type": None, "raw_text": "", "skip": False}

	soup = BeautifulSoup(html, "html.parser")
	text = soup.get_text(separator=" ", strip=True)

	# Exclude pre-packaged solutions robustly
	lowered = text.lower()
	if "pre-packaged" in lowered or "pre packaged" in lowered:
		return {"test_type": None, "raw_text": "", "skip": True}

	match = re.search(r"Test\s*Type[:\s]*([A-Z, ]{1,6})", text, re.I)
	test_type = match.group(1).strip() if match else None
	return {"test_type": test_type, "raw_text": text, "skip": False}


def crawl_catalog(max_pages=25, throttle=0.5):
	html = fetch(BASE)
	items = parse_catalog_page(html)

	for page in range(2, max_pages + 1):
		page_url = f"{BASE}?page={page}"
		html_page = fetch(page_url)
		if not html_page:
			break

		more_items = parse_catalog_page(html_page)
		if not more_items:
			break

		items.extend(more_items)
		time.sleep(throttle)

	df = pd.DataFrame(items).drop_duplicates(subset=["url"])
	print(f"Found {len(df)} product URLs")
	return df


def enrich_and_save(df, throttle=0.4):
	records = []

	for idx, row in df.iterrows():
		print(f"[{idx + 1}/{len(df)}] {row['title']}")
		meta = extract_product_page(row["url"])

		if meta.get("skip", False):
			print("  -> Skipping (pre-packaged)")
			continue

		records.append(
			{
				"title": row["title"],
				"url": row["url"],
				"test_type": meta["test_type"],
				"raw_text": meta["raw_text"],
			}
		)
		time.sleep(throttle)

	output_df = pd.DataFrame(records)
	output_df.to_csv("data/assessments.csv", index=False)
	print("Saved data/assessments.csv")
	return output_df


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Crawl SHL catalog")
	parser.add_argument("--max-pages", type=int, default=25, help="Number of catalog pages to crawl")
	parser.add_argument("--throttle", type=float, default=0.5, help="Seconds to sleep between requests")
	arguments = parser.parse_args()

	dataframe = crawl_catalog(max_pages=arguments.max_pages, throttle=arguments.throttle)
	if not dataframe.empty:
		enrich_and_save(dataframe, throttle=arguments.throttle)

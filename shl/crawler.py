# shl/crawler.py
"""
Full SHL Product Catalog Crawler
Scrapes all 'Individual Test Solutions' except 'Pre-packaged Job Solutions'
Outputs: shl/data/assessments.csv (or custom path via --output)
Usage: python crawler.py [--output PATH] [--pages N]
"""

import requests, re, csv, os, time, argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUT = os.path.join(os.path.dirname(__file__), "data", "assessments.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"}
THROTTLE = 0.7  # seconds between requests
MAX_PAGES = 32  # 32 pages per SHL catalog


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_links_from_page(page_url):
    soup = get_soup(page_url)
    links = []
    for a in soup.select("a[href*='/products/']"):
        href = a["href"]
        if "/products/" in href and "pre-packaged-job-solutions" not in href.lower():
            links.append(urljoin(BASE, href.split("?")[0]))
    return list(set(links))

def extract_product_details(url):
    soup = get_soup(url)
    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
    text = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
    text = re.sub(r"\s+", " ", text)
    
    # Skip invalid pages: empty title, category pages, pre-packaged solutions
    if not title or len(title) < 3:
        return None
    if "pre-packaged" in title.lower() or "job solution" in title.lower():
        return None
    # Skip category/overview pages (too generic URLs)
    if url.endswith("/assessments/") or url.endswith("/products/"):
        return None
    
    return {
        "title": title,
        "url": url,
        "raw_text": text,
        "test_type": detect_type(title + " " + text),
        "text": text,
    }

def detect_type(text):
    t = text.lower()
    if re.search(r"coding|developer|programming|sql|java|python|technical", t):
        return "technical"
    elif re.search(r"personality|behavior|behaviour|motivation|communication", t):
        return "behavioural"
    elif re.search(r"reasoning|cognitive|verbal|numerical|aptitude", t):
        return "cognitive"
    elif re.search(r"simulation|call center|sales|customer", t):
        return "simulation"
    else:
        return "general"

def crawl_all():
    all_links = []
    for i in range(0, MAX_PAGES * 12, 12):
        url = f"{CATALOG_URL}?start={i}&type=1"
        print(f"[+] Scanning: {url}")
        try:
            links = extract_links_from_page(url)
            all_links.extend(links)
        except Exception as e:
            print(f"Failed page {i}: {e}")
        time.sleep(THROTTLE)
    all_links = list(set(all_links))
    print(f"Total unique product links: {len(all_links)}")

    results = []
    for idx, link in enumerate(all_links, 1):
        try:
            data = extract_product_details(link)
            if data:
                results.append(data)
                print(f"[{idx}/{len(all_links)}] Added: {data['title'][:60]}")
        except Exception as e:
            print(f"Failed product: {link} -> {e}")
        time.sleep(THROTTLE)
    return results

def write_csv(rows, output_path=None):
    out_file = output_path or OUT
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "test_type", "raw_text", "text"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n✅ Saved {len(rows)} assessments to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl SHL product catalog")
    parser.add_argument("--output", "-o", help="Output CSV path (default: shl/data/assessments.csv)")
    parser.add_argument("--pages", "-p", type=int, help=f"Max pages to crawl (default: {MAX_PAGES})")
    args = parser.parse_args()
    
    # Override globals if specified
    if args.pages:
        MAX_PAGES = args.pages
        print(f"Crawling {MAX_PAGES} pages...")
    
    data = crawl_all()
    write_csv(data, args.output)

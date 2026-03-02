"""
scraper.py — Scrapes SHL Individual Test Solutions from the product catalog.
Only scrapes "Individual Test Solutions" (type=1), NOT Pre-packaged Job Solutions.
Saves results to ../data/shl_catalog.json
"""
import asyncio
import json
import os
import re
from pathlib import Path

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "shl_catalog.json"


async def scrape_all():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    if OUTPUT_PATH.exists():
        try:
            data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            if len(data) >= 300:
                print(f"Loaded {len(data)} assessments from cache.")
                return data
        except Exception:
            pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        assessment_links = []
        start = 0
        max_pages = 50  # safety limit

        for _ in range(max_pages):
            url = f"{CATALOG_URL}?start={start}&type=1"
            print(f"Fetching catalog page: {url}")
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(1000)
            except Exception as e:
                print(f"  Warning navigating {url}: {e}")

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            # Find all links in the table rows
            links_found = []

            # Try different selectors
            rows = soup.select("table tr")
            for row in rows:
                for a in row.find_all("a", href=True):
                    href = a["href"]
                    if "/product-catalog/view/" in href:
                        full_url = BASE_URL + href if href.startswith("/") else href
                        if full_url not in assessment_links and full_url not in links_found:
                            links_found.append(full_url)

            # Also check generic anchor tags containing /product-catalog/view/
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/product-catalog/view/" in href:
                    full_url = BASE_URL + href if href.startswith("/") else href
                    if full_url not in assessment_links and full_url not in links_found:
                        links_found.append(full_url)

            if not links_found:
                print(f"  No links found at start={start}, stopping pagination.")
                break

            assessment_links.extend(links_found)
            print(f"  Found {len(links_found)} links (total so far: {len(assessment_links)})")

            # Check for next page button
            next_btn = soup.select_one("a[rel='next']") or soup.select_one(".pagination__item--next a") or soup.select_one("li.next a")
            if next_btn:
                start += 12
            else:
                # Try checking if there are more pages by seeing if page had results
                # SHL uses ?start=0,12,24...
                start += 12
                # If we got less results than a full page, we might be done
                if len(links_found) < 3:
                    print("  Few results on this page, trying one more page then stopping.")
                    # Try one more
                    url2 = f"{CATALOG_URL}?start={start}&type=1"
                    try:
                        await page.goto(url2, wait_until="networkidle", timeout=30000)
                        await page.wait_for_timeout(500)
                    except Exception:
                        pass
                    content2 = await page.content()
                    soup2 = BeautifulSoup(content2, "html.parser")
                    any_links = [a["href"] for a in soup2.find_all("a", href=True) if "/product-catalog/view/" in a["href"]]
                    if not any_links:
                        break

        print(f"\nTotal assessment links found: {len(assessment_links)}")

        # De-duplicate
        assessment_links = list(dict.fromkeys(assessment_links))
        print(f"After deduplication: {len(assessment_links)}")

        # Scrape each detail page
        assessments = []
        for i, url in enumerate(assessment_links):
            try:
                detail = await scrape_detail(page, url)
                assessments.append(detail)
                if (i + 1) % 20 == 0:
                    print(f"Progress: {i+1}/{len(assessment_links)} scraped")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        await browser.close()

    print(f"\nSuccessfully scraped {len(assessments)} assessments")

    # Validate count
    if len(assessments) < 50:
        raise ValueError(
            f"Only scraped {len(assessments)} assessments — something went wrong with the scraper!"
        )

    OUTPUT_PATH.write_text(json.dumps(assessments, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(assessments)} assessments to {OUTPUT_PATH}")
    return assessments


async def scrape_detail(page, url: str) -> dict:
    """Scrape a single assessment detail page."""
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(500)
    except Exception as e:
        print(f"  Timeout/error loading {url}: {e}")

    content = await page.content()
    soup = BeautifulSoup(content, "html.parser")

    # Name
    name_el = soup.select_one("h1") or soup.select_one(".hero__title") or soup.select_one(".page-title")
    name = name_el.get_text(strip=True) if name_el else "Unknown"

    # Description — try multiple selectors
    desc_el = (
        soup.select_one(".product-catalogue-training-calendar__row--desc")
        or soup.select_one(".hero__intro")
        or soup.select_one(".product__description")
        or soup.select_one("section.intro p")
        or soup.select_one("article p")
        or soup.select_one(".content-block p")
    )
    description = desc_el.get_text(strip=True) if desc_el else ""

    # If description is empty, grab first meaningful paragraph
    if not description:
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 50:
                description = text[:500]
                break

    # Test types — look for badges/labels
    test_types = []
    for selector in [
        ".product-catalogue__key",
        ".badge",
        ".tag",
        ".label--type",
        "[class*='test-type']",
        "[class*='product-type']",
    ]:
        for badge in soup.select(selector):
            t = badge.get_text(strip=True)
            if t and len(t) < 60 and t not in test_types:
                test_types.append(t)

    # Fallback: look for known test type strings in the page text
    full_text = soup.get_text()
    known_types = [
        "Knowledge & Skills",
        "Personality & Behavior",
        "Personality & Behaviour",
        "Ability & Aptitude",
        "Competencies",
        "Biodata & Situational Judgement",
        "Simulations",
        "Development & 360",
        "Assessment Exercises",
    ]
    if not test_types:
        for kt in known_types:
            if kt.lower() in full_text.lower():
                test_types.append(kt)

    # Duration — parse minutes
    duration = 0
    duration_patterns = [
        r'(\d+)\s*(?:to\s*\d+\s*)?minutes?',
        r'approximately\s*(\d+)',
        r'(\d+)\s*min',
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            duration = int(match.group(1))
            break

    # Remote & Adaptive support
    full_text_lower = full_text.lower()
    remote_support = "Yes" if any(k in full_text_lower for k in ["remote testing", "remote proctoring", "remotely administered", "remote work"]) else "No"
    adaptive_support = "Yes" if any(k in full_text_lower for k in ["adaptive", "irt-based", "computer adaptive"]) else "No"

    return {
        "name": name,
        "url": url,
        "description": description[:1000],
        "test_type": test_types[:5],
        "duration": duration,
        "remote_support": remote_support,
        "adaptive_support": adaptive_support,
    }


if __name__ == "__main__":
    asyncio.run(scrape_all())

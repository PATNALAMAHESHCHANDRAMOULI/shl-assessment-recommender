"""
recommender.py — Hybrid vector search + Gemini LLM re-ranking for SHL assessments.
"""
import json
import os
import re

import google.generativeai as genai
import requests
from dotenv import load_dotenv

from embedder import search

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY and GOOGLE_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GOOGLE_API_KEY)
    _gemini_available = True
else:
    _gemini_available = False

_gemini_model = None


def _get_gemini():
    global _gemini_model
    if _gemini_model is None and _gemini_available:
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    return _gemini_model


def fetch_url_content(url: str) -> str:
    """Fetch and clean content from a job posting URL."""
    try:
        from bs4 import BeautifulSoup
        res = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        print(f"Warning: could not fetch URL {url}: {e}")
        return url


def _parse_duration_constraint(query: str):
    """Extract max/min duration from query text."""
    max_dur = None
    min_dur = None

    # "max X minutes", "within X minutes", "X minutes max", "no more than X"
    max_patterns = [
        r'(?:max(?:imum)?|within|no more than|up to|less than|under|at most)\s*(?:of\s*)?(\d+)\s*min',
        r'(\d+)\s*min(?:utes?)?\s*(?:max(?:imum)?|or less)',
        r'completed in\s*(\d+)\s*min',
    ]
    min_patterns = [
        r'(?:at least|minimum|more than|longer than|at least)\s*(\d+)\s*min',
        r'(\d+)\s*min(?:utes?)?\s*(?:long|minimum)',
    ]

    for pat in max_patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            max_dur = int(m.group(1))
            break

    for pat in min_patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            min_dur = int(m.group(1))
            break

    # Fallback: "X minutes" with "hour"
    if max_dur is None:
        hour_match = re.search(r'(\d+)\s*hour', query, re.IGNORECASE)
        if hour_match:
            max_dur = int(hour_match.group(1)) * 60

    return min_dur, max_dur


def recommend(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns up to top_k recommended SHL assessments for the given query.
    Pipeline: URL fetch → vector search → duration filter → Gemini re-rank → fallback
    """
    effective_query = query.strip()

    # If it's a URL, fetch its content
    if re.match(r'https?://', effective_query):
        print(f"Fetching content from URL: {effective_query}")
        effective_query = fetch_url_content(effective_query)

    # Parse duration constraints
    min_dur, max_dur = _parse_duration_constraint(query)
    print(f"Duration constraints — min: {min_dur}, max: {max_dur}")

    # Vector search — retrieve 30 candidates
    candidates = search(effective_query, n_results=30)

    # Apply duration filter (if constraint specified)
    if max_dur is not None:
        filtered = [c for c in candidates if c["duration"] == 0 or c["duration"] <= max_dur]
        if len(filtered) >= 5:
            candidates = filtered
        else:
            print(f"Duration filter too strict ({max_dur} min), keeping top {len(filtered)} filtered + extras")
            # Soft filter: prefer within range but don't drop below 5
            candidates = filtered + [c for c in candidates if c not in filtered]

    if min_dur is not None:
        filtered = [c for c in candidates if c["duration"] == 0 or c["duration"] >= min_dur]
        if len(filtered) >= 5:
            candidates = filtered

    # Gemini re-ranking
    gemini = _get_gemini()
    if gemini:
        candidates_json = json.dumps(
            [
                {
                    "url": c["url"],
                    "name": c["name"],
                    "test_type": c["test_type"],
                    "duration": c["duration"],
                    "description": (c.get("description") or "")[:250],
                    "remote_support": c["remote_support"],
                    "adaptive_support": c["adaptive_support"],
                }
                for c in candidates
            ],
            indent=2,
        )

        prompt = f"""You are an expert talent assessment consultant at SHL.
Given the hiring query or job description below, select the most relevant SHL assessments.

QUERY:
{effective_query[:2500]}

CANDIDATE ASSESSMENTS (JSON):
{candidates_json}

SELECTION RULES:
1. Select exactly 5 to 10 most relevant assessments.
2. If the query requires BOTH technical AND soft/behavioural skills, BALANCE across test types:
   - Knowledge & Skills, Ability & Aptitude, Personality & Behavior
3. Strictly respect any duration constraints mentioned (max/min minutes).
4. Prioritize assessments that directly match the role, skills, and seniority level.
5. Return ONLY a valid JSON array of URLs with no explanation, no markdown, no code fences.

Output format example: ["url1", "url2", "url3"]"""

        try:
            response = gemini.generate_content(prompt)
            text = response.text.strip()
            # Extract JSON array robustly
            match = re.search(r'\[[\s\S]*?\]', text)
            if match:
                selected_urls = json.loads(match.group())
            else:
                print("Gemini returned no JSON array, using vector results directly.")
                selected_urls = [c["url"] for c in candidates[:top_k]]
        except Exception as e:
            print(f"Gemini error (falling back to vector results): {e}")
            selected_urls = [c["url"] for c in candidates[:top_k]]
    else:
        print("Gemini not configured — using vector search results directly.")
        selected_urls = [c["url"] for c in candidates[:top_k]]

    # Map selected URLs back to full metadata
    url_map = {c["url"]: c for c in candidates}
    results = []
    seen_urls = set()

    for url in selected_urls:
        if url in url_map and url not in seen_urls:
            results.append(url_map[url])
            seen_urls.add(url)
        if len(results) >= top_k:
            break

    # Ensure minimum 5 results by filling from vector search
    if len(results) < 5:
        for c in candidates:
            if c["url"] not in seen_urls:
                results.append(c)
                seen_urls.add(c["url"])
            if len(results) >= 5:
                break

    return results[:top_k]

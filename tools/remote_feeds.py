"""
Remote Job Feed Fetchers – free, no-API-key-required data sources.
Fetches from RemoteOK and ArbeitNow and normalises their payloads into
dicts that map directly to the JobCache SQLite schema.

Academic Alignment: Module II/III – Informed Search (real data retrieval).
"""

import uuid
import logging
from typing import List, Dict, Optional

import requests

from tools.relevance import query_tokens, tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FAI-JobBot/1.0; +https://github.com/FAI-JobBot)"
    )
}
# (connect_timeout, read_timeout) – generous to handle slow public APIs
_TIMEOUT = (15, 25)
_MAX_RETRIES = 2  # number of fetch attempts before giving up


# ---------------------------------------------------------------------------
# Feed-side relevance
# ---------------------------------------------------------------------------
#
# Every feed prefilters client-side because most of these APIs either ignore the
# query or accept only a single tag. That filtering must use whole-word
# matching, otherwise it silently over-matches and the downstream ranker has to
# discard the noise.

def _feed_tokens(query: str) -> List[str]:
    """Query role tokens for feed-side prefiltering."""
    tokens = query_tokens(query)[0]
    return tokens or tokenize(query)


def _hits(text: str, tokens: List[str]) -> bool:
    """True when any query token appears as a whole word in *text*."""
    if not tokens:
        return False
    words = set(tokenize(text))
    return any(t in words for t in tokens)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_remoteok(item: dict) -> dict:
    """Map a RemoteOK job dict to our internal cache schema."""
    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "title": item.get("position") or item.get("title", ""),
        "company": item.get("company", ""),
        "location": item.get("location") or "Remote",
        "url": item.get("url", ""),
        "description": item.get("description", ""),
        "source": "RemoteOK",
        "employment_type": "Full-time",
        "required_skills": ",".join(item.get("tags", [])),
    }


def _normalise_arbeitnow(item: dict) -> dict:
    """Map an ArbeitNow job dict to our internal cache schema."""
    return {
        "id": str(item.get("slug") or uuid.uuid4()),
        "title": item.get("title", ""),
        "company": item.get("company_name", ""),
        "location": item.get("candidate_required_location") or "Remote",
        "url": item.get("url", ""),
        "description": item.get("description", ""),
        "source": "ArbeitNow",
        "employment_type": "Full-time" if not item.get("job_types") else ", ".join(item.get("job_types", [])),
        "required_skills": "",
    }


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_remoteok(query: str) -> List[Dict]:
    """
    Fetch jobs from RemoteOK (https://remoteok.com/api).
    No API key required – returns up to ~100 recent remote jobs and filters
    by query keyword present in position title or tags.
    Retries up to _MAX_RETRIES times on transient network failures.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(
                "https://remoteok.com/api",
                headers=_HEADERS,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            # First element is always metadata – skip it
            raw_jobs = data[1:] if isinstance(data, list) and len(data) > 1 else []

            tokens = _feed_tokens(query)
            matched = []
            for item in raw_jobs:
                if not isinstance(item, dict):
                    continue
                title = item.get("position") or item.get("title", "")
                tags = " ".join(item.get("tags", []) or [])
                # Whole-word match against title or tags. Substring matching here
                # let "ai" match "Training" and "ml" match "HTML", which is how
                # unrelated RemoteOK roles entered the result set.
                if _hits(title, tokens) or _hits(tags, tokens):
                    matched.append(_normalise_remoteok(item))

            logger.info(f"[RemoteOK] Matched {len(matched)} jobs for query='{query}'")
            return matched

        except Exception as exc:
            last_exc = exc
            logger.warning(f"[RemoteOK] Attempt {attempt}/{_MAX_RETRIES} failed: {exc}")

    logger.warning(f"[RemoteOK] All {_MAX_RETRIES} attempts failed. Skipping source.")
    return []


def fetch_arbeitnow(query: str) -> List[Dict]:
    """
    Fetch jobs from ArbeitNow (https://www.arbeitnow.com/api/job-board-api).
    Free, no API key required – returns remote-friendly listings.
    """
    try:
        resp = requests.get(
            "https://www.arbeitnow.com/api/job-board-api",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()

        raw_jobs = payload.get("data", [])
        tokens = _feed_tokens(query)
        matched = []
        for item in raw_jobs:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            # Title only. Matching the description let any posting whose
            # boilerplate mentioned the field count as a hit.
            if _hits(title, tokens):
                matched.append(_normalise_arbeitnow(item))

        logger.info(f"[ArbeitNow] Matched {len(matched)} jobs for query='{query}'")
        return matched

    except Exception as exc:
        logger.warning(f"[ArbeitNow] Fetch failed: {exc}")
        return []


def _normalise_remotive(item: dict) -> dict:
    """Map a Remotive job dict to our internal cache schema."""
    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "title": item.get("title", ""),
        "company": item.get("company_name", ""),
        "location": item.get("candidate_required_location") or "Remote",
        "url": item.get("url", ""),
        "description": item.get("description", ""),
        "source": "Remotive",
        "employment_type": item.get("job_type") or "Full-time",
        "required_skills": ",".join(item.get("tags", [])),
    }


def fetch_remotive(query: str) -> List[Dict]:
    """
    Fetch jobs from Remotive's official public API (https://remotive.com/api/remote-jobs).
    Free, no API key required.
    """
    try:
        resp = requests.get(
            f"https://remotive.com/api/remote-jobs?search={requests.utils.quote(query)}&limit=25",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_jobs = data.get("jobs", [])
        tokens = _feed_tokens(query)
        matched = []
        for item in raw_jobs:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or ""
            tags = " ".join(item.get("tags") or [])
            # Title or tags. The previous code built the tag string but matched
            # on the title twice, so tag evidence never actually counted.
            if _hits(title, tokens) or _hits(tags, tokens):
                matched.append(_normalise_remotive(item))

        logger.info(f"[Remotive] Matched {len(matched)} relevant jobs for query='{query}'")
        return matched
    except Exception as exc:
        logger.warning(f"[Remotive] Fetch failed: {exc}")
        return []


def _normalise_jobicy(item: dict) -> dict:
    """Map a Jobicy job dict to our internal cache schema."""
    # Jobicy returns "jobType" as either a string ("Full-Time") or a list
    # (["Full-Time"]) depending on the record, so flatten it defensively.
    raw_type = item.get("jobType")
    if isinstance(raw_type, (list, tuple, set)):
        employment_type = ", ".join(str(t) for t in raw_type if t) or "Full-time"
    else:
        employment_type = str(raw_type) if raw_type else "Full-time"

    raw_geo = item.get("jobGeo")
    if isinstance(raw_geo, (list, tuple, set)):
        location = ", ".join(str(g) for g in raw_geo if g) or "Remote"
    else:
        location = str(raw_geo) if raw_geo else "Remote"

    return {
        "id": str(item.get("id") or uuid.uuid4()),
        "title": item.get("jobTitle", ""),
        "company": item.get("companyName", ""),
        "location": location,
        "url": item.get("url", ""),
        "description": item.get("jobDescription", ""),
        "source": "Jobicy",
        "employment_type": employment_type,
        "required_skills": ",".join(item.get("jobIndustry", []) if isinstance(item.get("jobIndustry"), list) else []),
    }


def fetch_jobicy(query: str) -> List[Dict]:
    """
    Fetch jobs from Jobicy public API (https://jobicy.com/api/v2/remote-jobs).
    Free, no API key required.
    """
    try:
        resp = requests.get(
            f"https://jobicy.com/api/v2/remote-jobs?count=25&tag={requests.utils.quote(query)}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            raw_jobs = data.get("jobs", [])
            tokens = _feed_tokens(query)
            matched = []
            for item in raw_jobs:
                if not isinstance(item, dict):
                    continue
                title = item.get("jobTitle") or ""
                # Title only; the API's `tag` param already does the coarse
                # server-side narrowing, so the description adds only noise.
                if _hits(title, tokens):
                    matched.append(_normalise_jobicy(item))
            logger.info(f"[Jobicy] Matched {len(matched)} relevant jobs for query='{query}'")
            return matched
        return []
    except Exception as exc:
        logger.warning(f"[Jobicy] Fetch failed: {exc}")
        return []


def fetch_linkedin(query: str, location: str = "") -> List[Dict]:
    """
    Scrape public guest job postings from LinkedIn without requiring authentication.
    Endpoint: https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
    """
    try:
        # LinkedIn returns loosely-related results for a keywords search, and
        # the guest endpoint is location-biased toward the requester's own
        # region when no location is given. Both had to be compensated for
        # client-side below.
        keywords = " ".join(_feed_tokens(query)[:4]) or query
        encoded_query = requests.utils.quote(keywords)
        loc = requests.utils.quote(location) if location and location.lower() != "remote" else ""
        url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={encoded_query}&start=0"
        if loc:
            url += f"&location={loc}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"[LinkedIn] Returned status {resp.status_code}")
            return []

        # Parse HTML response using BeautifulSoup or regex fallback
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            job_cards = soup.find_all("li")
        except ImportError:
            soup = None
            job_cards = []

        jobs = []
        if soup and job_cards:
            for card in job_cards:
                title_elem = card.find("h3", class_="base-search-card__title")
                company_elem = card.find("h4", class_="base-search-card__subtitle")
                location_elem = card.find("span", class_="job-search-card__location")
                link_elem = card.find("a", class_="base-card__full-link") or card.find("a")

                if not title_elem or not link_elem:
                    continue

                title = title_elem.get_text(strip=True)
                company = company_elem.get_text(strip=True) if company_elem else "Company"
                loc_text = location_elem.get_text(strip=True) if location_elem else (location or "Remote")
                job_url = link_elem.get("href", "").split("?")[0]

                jobs.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "company": company,
                    "location": loc_text,
                    "url": job_url,
                    "description": f"Role: {title} at {company}. View full details on LinkedIn.",
                    "source": "LinkedIn",
                    "employment_type": "Full-time",
                    "required_skills": "",
                })
        else:
            # Simple regex fallback if BeautifulSoup is not yet installed
            import re
            titles = re.findall(r'<h3 class="base-search-card__title">\s*([^<]+)\s*</h3>', resp.text)
            companies = re.findall(r'<h4 class="base-search-card__subtitle">[^<]*<a[^>]*>\s*([^<]+)\s*</a>', resp.text)
            links = re.findall(r'<a class="base-card__full-link"[^>]*href="([^"]+)"', resp.text)
            
            for i in range(min(len(titles), len(links))):
                title = titles[i].strip()
                company = companies[i].strip() if i < len(companies) else "Company"
                job_url = links[i].split("?")[0]
                jobs.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "company": company,
                    "location": location or "Remote",
                    "url": job_url,
                    "description": f"Role: {title} at {company}. View full details on LinkedIn.",
                    "source": "LinkedIn",
                    "employment_type": "Full-time",
                    "required_skills": "",
                })

        # Client-side filter. LinkedIn's guest search happily returns loosely
        # related cards, and every card is given a synthetic description that
        # contains the title verbatim — so the downstream ranker, which reads
        # that description, cannot tell these apart from genuine keyword hits.
        # Filtering here is the only place it can be caught.
        tokens = _feed_tokens(query)
        if tokens:
            jobs = [j for j in jobs if _hits(j.get("title", ""), tokens)]

        logger.info(f"[LinkedIn] Scraped {len(jobs)} on-target jobs for query='{query}'")
        return jobs
    except Exception as exc:
        logger.warning(f"[LinkedIn] Scraping failed: {exc}")
        return []


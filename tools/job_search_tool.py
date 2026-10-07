"""
Job search tool – Hybrid backend: public RSS/JSON feeds + local SQLite cache.

Strategy (in priority order):
  1. Return recent results from the local SQLite cache (< 24 h old).
  2. If the cache is empty / stale, fetch from free public feeds:
       - RemoteOK  (https://remoteok.com/api)
       - ArbeitNow (https://www.arbeitnow.com/api/job-board-api)
  3. Normalise feed payloads → JobListing schema.
  4. Persist all fresh results back into SQLite for future calls.

Academic Alignment: Modules II & III (Uninformed & Informed Search Strategies).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from config.settings import settings
from schemas.job_schema import JobListing
from tools.job_cache import JobCache
from tools.relevance import filter_and_rank, query_tokens
from tools.remote_feeds import (
    fetch_remoteok,
    fetch_arbeitnow,
    fetch_remotive,
    fetch_jobicy,
    fetch_linkedin,
)

logger = logging.getLogger(__name__)


_SENIORITY_LABELS = (
    ("intern", 0), ("trainee", 0), ("apprentice", 0),
    ("entry", 1), ("fresher", 1), ("fresh", 1), ("junior", 1),
    ("graduate", 1), ("associate", 1),
    ("mid", 2), ("intermediate", 2),
    ("senior", 3), ("lead", 4), ("staff", 4), ("principal", 5),
    ("head", 5), ("director", 6),
)


def _rank_from_label(experience_level: str) -> Optional[int]:
    """Map a free-text seniority label ('Entry-level', 'Junior') to a ladder rank."""
    lowered = (experience_level or "").lower()
    for label, rank in _SENIORITY_LABELS:
        if label in lowered:
            return rank
    return None


def _dict_to_job_listing(raw: dict, fallback_query: str = "") -> Optional[JobListing]:
    """
    Convert a normalised cache/feed dict to a JobListing Pydantic model.
    Returns None if the dict is missing mandatory fields.
    """
    job_id = raw.get("id", "")
    title = raw.get("title", "").strip()
    company = raw.get("company", "").strip()
    url = raw.get("url", "").strip()
    description = raw.get("description", "").strip()

    if not job_id or not title:
        return None

    # Parse comma-separated required_skills stored in cache
    skills_raw = raw.get("required_skills", "") or ""
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

    return JobListing(
        id=job_id,
        title=title or fallback_query,
        company=company or "Unknown Company",
        location=raw.get("location") or "Remote",
        employment_type=raw.get("employment_type") or "Full-time",
        description=description or "No description available.",
        apply_link=url or "https://remoteok.com",
        portal_name=raw.get("source") or "Public Feed",
        required_skills=skills,
    )


class JobSearchTool:

    def __init__(self):
        self._cache = JobCache(settings.SQLITE_DB_PATH)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_jobs(
        self,
        query: str,
        location: str = "Remote",
        experience_level: str = "Entry-level",
        max_results: int = 10,
        max_age_hours: int = 24,
        force_refresh: bool = False,
        seniority_rank: Optional[int] = None,
    ) -> List[JobListing]:
        """
        Return job listings for the given query.

        Args:
            query:            Role / keyword search string.
            location:         Preferred location (used for display only with
                              free feeds – they are inherently remote-focused).
            experience_level: Seniority hint. When *seniority_rank* is omitted it
                              is derived from this string.
            max_results:      Maximum number of JobListing objects to return.
            max_age_hours:    How far back in the cache to look (default 24 h).
                              Pass ``120`` to include jobs from the last 5 days.
            force_refresh:    Skip cache check and always pull from feeds.
            seniority_rank:   Target ladder rank (see relevance.extract_seniority).
                              Listings materially above it are demoted.

        Returns:
            List[JobListing] – empty when nothing matches the query. Callers must
            widen the query rather than treating an empty result as an error.
        """
        logger.info(
            f"[JobSearchTool] search_jobs(query='{query}', "
            f"max_age_hours={max_age_hours}, force_refresh={force_refresh})"
        )

        tokens, query_rank, _ = query_tokens(query)
        if seniority_rank is None:
            seniority_rank = query_rank if query_rank is not None else _rank_from_label(
                experience_level
            )

        # Over-fetch from the cache: the cache cannot rank, so asking it for
        # exactly max_results would let an unranked head-of-queue row crowd out
        # a genuinely on-target listing sitting further down.
        candidate_limit = max(max_results * 4, 40)

        # ── 1. Try cache (unless caller wants a forced refresh) ──────────────
        if not force_refresh:
            cached_raw = self._cache.get_recent(
                query,
                location=location,
                max_age_hours=max_age_hours,
                limit=candidate_limit,
            )
            if cached_raw:
                logger.info(f"[JobSearchTool] Cache hit → {len(cached_raw)} records")
                listings = self._raw_list_to_listings(
                    cached_raw, query, candidate_limit, seniority_rank
                )
                # Only trust the cache when it actually holds on-target results.
                # An empty ranking means "no match", not "fall back to junk".
                if listings:
                    return listings[:max_results]

        # ── 2. Fetch from public feeds ────────────────────────────────────────
        logger.info("[JobSearchTool] Cache miss – fetching from public feeds…")
        fresh_raw = self._fetch_from_feeds(query, location=location)

        if not fresh_raw:
            logger.warning("[JobSearchTool] No results from any feed source.")
            return []

        # ── 3. Persist to cache ───────────────────────────────────────────────
        written = self._cache.bulk_upsert(fresh_raw)
        logger.info(f"[JobSearchTool] Stored {written} jobs in SQLite cache")

        # Rank the union of fresh + cached rows so a cold cache and a warm one
        # cannot return different results for the same query.
        combined = fresh_raw + (cached_raw if not force_refresh else [])
        return self._raw_list_to_listings(
            combined, query, candidate_limit, seniority_rank
        )[:max_results]

    def refresh_cache(self, query: str, location: str = "Remote", max_results: int = 10) -> List[JobListing]:
        """Force-refresh: always hit feeds, ignore cached data."""
        return self.search_jobs(query, location=location, max_results=max_results, force_refresh=True)

    def search_jobs_by_date_range(
        self,
        query: str,
        from_dt: datetime,
        to_dt: Optional[datetime] = None,
        max_results: int = 50,
    ) -> List[JobListing]:
        """
        Return cached jobs whose ``fetched_at`` falls within [from_dt, to_dt].

        This queries **only the local SQLite cache** – it will NOT trigger a
        remote feed fetch.  Call :meth:`refresh_cache` first if you want to
        pull the latest data before filtering by date.

        Args:
            query:       Keyword fuzzy-matched on title / company / location.
            from_dt:     Window start (UTC datetime).
            to_dt:       Window end; defaults to now when omitted.
            max_results: Maximum results to return.

        Example – everything fetched in the last 5 days::

            from datetime import datetime, timedelta, timezone
            tool.search_jobs_by_date_range(
                "Python Engineer",
                from_dt=datetime.now(timezone.utc) - timedelta(days=5),
            )
        """
        if to_dt is None:
            to_dt = datetime.now(timezone.utc)
        raw = self._cache.get_by_date_range(query, from_dt=from_dt, to_dt=to_dt, limit=max_results)
        return self._raw_list_to_listings(raw, query, max_results)

    def cache_stats(self) -> dict:
        """Return basic statistics about the local job cache."""
        return {"total_cached_jobs": self._cache.count(), "db_path": str(self._cache.db_path)}


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_feeds(self, query: str, location: str = "Remote") -> List[dict]:
        """
        Query all registered public feed and scraper sources and combine results:
          1. LinkedIn  (guest public search scraping - location-aware)
          2. Remotive  (free remote jobs API)
          3. Jobicy    (free remote tech jobs API)
          4. RemoteOK  (free remote developer feed)
          5. ArbeitNow (free job board API)

        The sources are fetched concurrently rather than one after another.
        Each carries a (15, 25) connect/read timeout, so the sequential version
        could spend well over a minute just waiting on a single slow feed; in
        parallel the whole batch costs roughly as much as the slowest one.

        Each source's errors are still caught independently so one failure does
        not block the others, and results are collected in the declared source
        order so downstream relevance ranking stays deterministic.
        """
        sources = [
            ("LinkedIn", lambda q: fetch_linkedin(q, location=location)),
            ("Remotive", fetch_remotive),
            ("Jobicy", fetch_jobicy),
            ("RemoteOK", fetch_remoteok),
            ("ArbeitNow", fetch_arbeitnow),
        ]

        def run_one(name: str, fn) -> Tuple[str, List[dict]]:
            try:
                return name, fn(query) or []
            except Exception as exc:
                logger.warning(f"[JobSearchTool] {name} failed: {exc}")
                return name, []

        combined: List[dict] = []
        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = [pool.submit(run_one, n, f) for n, f in sources]
            # Preserve declaration order; only the wall-clock overlaps.
            for future in futures:
                name, results = future.result()
                logger.info(f"[JobSearchTool] {name} → {len(results)} results")
                combined.extend(results)

        return combined

    def _raw_list_to_listings(
        self,
        raw_list: List[dict],
        query: str,
        max_results: int,
        seniority_rank: Optional[int] = None,
    ) -> List[JobListing]:
        """
        Convert raw feed/cache dicts to JobListing objects, ranked by relevance.

        Free feeds are unordered and remote-first, so the first N records are
        frequently off-target (e.g. "Backend Engineer" returning React roles).
        Every candidate is scored against the query and only listings whose
        *title* corroborates the query survive.

        A query with no usable role tokens (e.g. the user typed only "Remote")
        carries no signal to filter on, so feed order is preserved in that one
        case. Otherwise an empty result means "no match" and is returned as such
        — the caller widens the query rather than showing unrelated roles.
        """
        listings: List[JobListing] = []
        for raw in raw_list:
            listing = _dict_to_job_listing(raw, fallback_query=query)
            if listing is not None:
                listings.append(listing)

        if not listings:
            return []

        tokens, _, _ = query_tokens(query)
        if not tokens:
            logger.info(
                "[JobSearchTool] Query %r has no role tokens – returning feed order.",
                query,
            )
            return listings[:max_results]

        ranked = filter_and_rank(
            listings, tokens, target_rank=seniority_rank, limit=max_results
        )
        if len(ranked) < len(listings):
            logger.info(
                "[JobSearchTool] Relevance filter kept %d/%d listings for %r.",
                len(ranked), len(listings), query,
            )
        if not ranked:
            logger.warning(
                "[JobSearchTool] No listing title matched query %r (tokens=%s).",
                query, tokens,
            )
        return ranked


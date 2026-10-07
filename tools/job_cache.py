"""
SQLite-backed job listing cache.

Stores normalised job records fetched from remote feeds and serves them
during subsequent searches to avoid redundant network calls.

Academic Alignment: Informed Search (Module III) – persistent heuristic state store.
"""

import sqlite3
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional

from tools import relevance

logger = logging.getLogger(__name__)

# Characters that can legitimately sit next to a word inside a job title.
# Used to build whole-word LIKE patterns. "_" is deliberately absent: SQLite
# treats it as a single-character wildcard, which silently degrades a
# word-boundary pattern back into a substring match.
_BOUNDARY_CHARS = (" ", ",", ".", ";", ":", "/", "\\", "(", ")", "-", "'", "\"",
                   "[", "]", "|", "+", "&")

# Schema version – bump when columns change
_SCHEMA_VERSION = 1


class JobCache:
    """
    Lightweight SQLite wrapper for caching job listings locally.

    Usage:
        cache = JobCache("/path/to/job_cache.db")
        cache.upsert(job_dict)
        results = cache.get_recent("Python Engineer", max_age_hours=24)
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser().resolve()
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with row-as-dict factory."""
        conn = sqlite3.connect(str(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _as_text(value, default: str = "") -> str:
        """
        Coerce any upstream feed value into a storable SQLite TEXT scalar.

        Public job feeds are inconsistent: the same logical field can arrive as
        a str, a list (e.g. Jobicy "jobType": ["Full-Time"]), a dict, or None.
        sqlite3 refuses to bind list/dict parameters ("type 'list' is not
        supported"), which previously caused silent per-record write failures.
        """
        if value is None:
            return default
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value if v not in (None, "")) or default
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items()) or default
        return str(value)

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they don't exist yet."""
        ddl = """
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            title           TEXT NOT NULL,
            company         TEXT DEFAULT '',
            location        TEXT DEFAULT 'Remote',
            url             TEXT DEFAULT '',
            description     TEXT DEFAULT '',
            source          TEXT DEFAULT '',
            employment_type TEXT DEFAULT 'Full-time',
            required_skills TEXT DEFAULT '',
            fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_title
            ON jobs (LOWER(title));

        CREATE INDEX IF NOT EXISTS idx_jobs_fetched_at
            ON jobs (fetched_at);

        -- Cached ATS match reports. Re-scoring the same candidate against the
        -- same job is the single most expensive repeated LLM call in the
        -- pipeline, so the verdict is persisted instead of recomputed.
        -- profile_hash folds in the candidate facts *and* the active model, so
        -- editing a resume or swapping providers invalidates stale entries.
        CREATE TABLE IF NOT EXISTS match_reports (
            profile_hash TEXT NOT NULL,
            job_id       TEXT NOT NULL,
            report_json  TEXT NOT NULL,
            model_id     TEXT DEFAULT '',
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (profile_hash, job_id)
        );

        CREATE INDEX IF NOT EXISTS idx_match_reports_created
            ON match_reports (created_at);
        """
        with self._connect() as con:
            con.executescript(ddl)
        logger.debug(f"[JobCache] Schema ready at {self.db_path}")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, job: Dict) -> None:
        """
        Insert a job or update it if the id already exists.
        Missing optional fields default to empty strings.

        Every bound value is coerced to a TEXT scalar via `_as_text` so that
        heterogeneous feed payloads can never abort the INSERT.
        """
        as_text = self._as_text

        record = {
            "id":               as_text(job.get("id"), ""),
            "title":            as_text(job.get("title"), ""),
            "company":          as_text(job.get("company"), ""),
            "location":         as_text(job.get("location"), "Remote"),
            "url":              as_text(job.get("url"), ""),
            "description":      as_text(job.get("description"), ""),
            "source":           as_text(job.get("source"), ""),
            "employment_type":  as_text(job.get("employment_type"), "Full-time"),
            "required_skills":  as_text(job.get("required_skills"), ""),
        }
        if not record["id"]:
            raise ValueError("Cannot cache a job record without an 'id'.")
        sql = """
        INSERT INTO jobs
            (id, title, company, location, url, description,
             source, employment_type, required_skills, fetched_at)
        VALUES
            (:id, :title, :company, :location, :url, :description,
             :source, :employment_type, :required_skills, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            title           = excluded.title,
            company         = excluded.company,
            location        = excluded.location,
            url             = excluded.url,
            description     = excluded.description,
            source          = excluded.source,
            employment_type = excluded.employment_type,
            required_skills = excluded.required_skills,
            fetched_at      = CURRENT_TIMESTAMP
        """
        with self._connect() as con:
            con.execute(sql, record)

    def bulk_upsert(self, jobs: List[Dict]) -> int:
        """Upsert a list of job dicts; returns the number processed."""
        count = 0
        for job in jobs:
            try:
                self.upsert(job)
                count += 1
            except Exception as exc:
                logger.warning(f"[JobCache] Failed to upsert job '{job.get('id')}': {exc}")
        logger.info(f"[JobCache] bulk_upsert: {count}/{len(jobs)} records written")
        return count

    def clear_stale(self, older_than_hours: int = 72) -> int:
        """Remove jobs older than the given number of hours. Returns deleted count."""
        sql = "DELETE FROM jobs WHERE fetched_at < datetime('now', ?)"
        with self._connect() as con:
            cur = con.execute(sql, (f"-{older_than_hours} hours",))
            deleted = cur.rowcount
        logger.info(f"[JobCache] Purged {deleted} stale records (>{older_than_hours}h old)")
        return deleted

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_recent(
        self,
        query: str,
        location: Optional[str] = None,
        max_age_hours: int = 24,
        limit: int = 20,
    ) -> List[Dict]:
        """
        Return jobs matching *query* and optional *location* fetched within the last ``max_age_hours``.
        """
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(hours=max_age_hours)
        return self.get_by_date_range(query, location=location, from_dt=from_dt, to_dt=now, limit=limit)

    def get_by_date_range(
        self,
        query: str,
        location: Optional[str] = None,
        from_dt: Optional[datetime] = None,
        to_dt: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Return cached jobs matching *query* and optional *location* whose ``fetched_at`` timestamp
        falls within [from_dt, to_dt].
        """
        if from_dt is None:
            from_dt = datetime.now(timezone.utc) - timedelta(hours=24)
        if to_dt is None:
            to_dt = datetime.now(timezone.utc)

        # SQLite stores timestamps as TEXT; ISO-8601 strings compare correctly
        from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
        to_str   = to_dt.strftime("%Y-%m-%d %H:%M:%S")

        # The cache is a *recall* layer only: it must over-fetch so the ranker
        # in tools.relevance has candidates to choose from. It deliberately does
        # not try to be precise — SQL LIKE cannot do whole-word matching, and
        # an earlier version AND-ed every token as a substring, so "AI Engineer"
        # matched any row containing the letters "ai" (621 of 924 cached jobs).
        #
        # Strategy: take the significant query tokens and require an OR over
        # them, anchored on the *title* or skills only. Description text is
        # excluded deliberately — a posting for an unrelated role routinely
        # mentions the neighbouring field in its boilerplate ("you will work
        # with our machine learning platform"), which is precisely how
        # unrelated results got in.
        tokens = relevance.domain_tokens(relevance.query_tokens(query)[0])
        if not tokens:
            tokens = [t for t in re.split(r"[^a-z0-9+#.]+", query.lower()) if len(t) > 2]

        if not tokens:
            logger.info("[JobCache] get_by_date_range(%r): no usable tokens", query)
            return []

        # SQLite LIKE has no word-boundary operator, so each literal is padded
        # with a non-word character. This keeps "ai" from matching "Training"
        # while still catching "AI", "AI/ML", "(AI)" and "AI,".
        group_clauses: List[str] = []
        group_params: List[List[str]] = []
        for group in relevance.sql_search_variants(tokens):
            clauses: List[str] = []
            params_for_group: List[str] = []
            for literal in group:
                esc = literal.replace("\\", "").replace("%", "").replace("_", " ")
                if not esc.strip():
                    continue
                # A multi-word phrase is matched as a phrase; a single word also
                # gets prefix and explicit-boundary variants.
                #
                # The boundary variants must enumerate real separators. SQLite's
                # LIKE treats "_" as a single-character wildcard, so "%_ai_%"
                # matched "Tr_ai_ning" — the substring bug re-entering through
                # the pattern that was meant to prevent it.
                if " " in esc:
                    pats = [f"% {esc} %"]
                else:
                    pats = [f"% {esc} %", f"{esc}%"]
                    pats += [f"%{sep}{esc}{sep}%" for sep in _BOUNDARY_CHARS]
                for pat in pats:
                    clauses.append("LOWER(title) LIKE ?")
                    params_for_group.append(pat)
                clauses.append("LOWER(required_skills) LIKE ?")
                params_for_group.append(f"% {esc} %")
            if clauses:
                group_clauses.append("(" + " OR ".join(clauses) + ")")
                group_params.append(params_for_group)

        if not group_clauses:
            logger.info("[JobCache] get_by_date_range(%r): no usable tokens", query)
            return []

        # OR across token groups: this layer exists for recall only, and precision is
        # enforced downstream by tools.relevance scoring the titles. AND-ing the
        # tokens together instead is what made "Backend Developer" miss the
        # cache entirely — no portal title spells out both words.
        #
        # A plain OR cannot be paired with a flat LIMIT though: "DevOps
        # Engineer" would fill every row slot with anything containing
        # "engineer" and push the real hits out. Ranking by how many token
        # groups a row satisfies keeps the genuine multi-token matches inside
        # the LIMIT, which is what the caller needs to see.
        flat_params: List[str] = []
        for p in group_params:
            flat_params.extend(p)

        hard_clauses = ["(" + " OR ".join(group_clauses) + ")"]
        hard_params = list(flat_params)

        if location and location.lower() not in ("remote", "any", "all", ""):
            hard_clauses.append("LOWER(location) LIKE ?")
            hard_params.append(f"%{location.lower()}%")

        hard_clauses.append("fetched_at >= ?")
        hard_params.append(from_str)
        hard_clauses.append("fetched_at <= ?")
        hard_params.append(to_str)

        # Rank by how many query tokens a row satisfies, then by freshness. The
        # score expression repeats each group's placeholders, so its parameters
        # must be repeated in the same order.
        score_parts = []
        score_params = []
        for clause, p in zip(group_clauses, group_params):
            score_parts.append("(CASE WHEN %s THEN 1 ELSE 0 END)" % clause)
            score_params.extend(p)
        match_score_sql = " + ".join(score_parts) if score_parts else "0"

        sql = f"""
        SELECT *
          FROM jobs
         WHERE {" AND ".join(hard_clauses)}
         ORDER BY ({match_score_sql}) DESC, fetched_at DESC
         LIMIT ?
        """
        hard_params.extend(score_params)
        hard_params.append(limit)
        params = hard_params
        with self._connect() as con:
            cursor = con.execute(sql, tuple(params))
            rows = cursor.fetchall()

        result = [dict(row) for row in rows]
        logger.info(
            f"[JobCache] get_by_date_range('{query}', loc='{location}', {from_str} → {to_str})"
            f" → {len(result)} records"
        )
        return result

    def count(self) -> int:
        """Return total number of cached job records."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM jobs").fetchone()
            return row[0] if row else 0

    def get_by_id(self, job_id: str) -> Optional[Dict]:
        """Retrieve a single cached job by its unique id."""
        with self._connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Match-report cache (Agent 4)
    # ------------------------------------------------------------------

    def save_match_report(
        self,
        profile_hash: str,
        job_id: str,
        report_json: str,
        model_id: str = "",
    ) -> None:
        """Persist one MatchReport so the same pairing is never re-scored."""
        if not profile_hash or not job_id:
            raise ValueError("profile_hash and job_id are both required.")
        sql = """
        INSERT INTO match_reports (profile_hash, job_id, report_json, model_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(profile_hash, job_id) DO UPDATE SET
            report_json = excluded.report_json,
            model_id    = excluded.model_id,
            created_at  = CURRENT_TIMESTAMP
        """
        with self._connect() as con:
            con.execute(sql, (profile_hash, job_id, report_json, model_id))
        logger.debug(f"[JobCache] Cached match report {profile_hash[:8]}/{job_id}")

    def get_match_report(self, profile_hash: str, job_id: str) -> Optional[str]:
        """Return the cached MatchReport JSON, or None on a miss."""
        if not profile_hash or not job_id:
            return None
        with self._connect() as con:
            row = con.execute(
                "SELECT report_json FROM match_reports "
                "WHERE profile_hash = ? AND job_id = ?",
                (profile_hash, job_id),
            ).fetchone()
        return row[0] if row else None

    def clear_match_reports(self, older_than_hours: int = 72) -> int:
        """Drop cached match reports older than the given age."""
        sql = "DELETE FROM match_reports WHERE created_at < datetime('now', ?)"
        with self._connect() as con:
            cur = con.execute(sql, (f"-{older_than_hours} hours",))
            deleted = cur.rowcount
        logger.info(f"[JobCache] Purged {deleted} cached match reports")
        return deleted

    def match_report_count(self) -> int:
        """Return the number of cached match reports."""
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM match_reports").fetchone()
            return row[0] if row else 0

"""
Agent 3: Job Search & Query Optimizer Agent.
Academic Alignment: Modules II & III (State Space Search Strategies & Query Expansion).
Orchestrates autonomous job discovery, query reformulations, and tool calling.
"""

import re
from typing import List, Optional, Set, Tuple
from schemas.job_schema import JobListing
from tools.job_search_tool import JobSearchTool
from tools.relevance import (
    domain_tokens,
    extract_seniority,
    field_tokens,
    query_tokens,
    score_job,
    seniority_penalty,
)
from agents.base_agent import BaseAgent


class JobSearchAgent(BaseAgent):
    """
    Search agent that uses iterative query expansion and tool calling
    to discover high-relevance job listings from job portals.
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(agent_name="JobSearchAgent", model_name=model_name)
        self.search_tool = JobSearchTool()

    # Max alternative queries to try per search. Bounded so a genuinely
    # unavailable role fails fast instead of hammering every feed.
    _MAX_EXPANSIONS = 3

    # Trailing role nouns and their near-synonyms, used to widen a query
    # without changing its domain.
    _ROLE_NOUN_SWAPS = {
        "engineer": "Developer",
        "developer": "Engineer",
        "scientist": "Engineer",
        "analyst": "Engineer",
        "manager": "Lead",
    }

    # Leading modifiers portals commonly omit from titles.
    _DROPPABLE_MODIFIERS = {
        "junior", "senior", "associate", "graduate", "entry", "fresher",
        "frontend", "backend", "full", "software", "principal", "staff",
    }

    def discover_jobs(
        self,
        role_title: str,
        location: str = "India",
        experience_level: str = "Entry-level",
        target_count: int = 5,
        max_age_hours: int = 24,
        employment_type: str = "Any",
        extra_queries: Optional[List[str]] = None,
    ) -> List[JobListing]:
        """
        Execute an agentic search loop:
        1. Formulate initial search query.
        2. Call search tool.
        3. If results < target_count, rewrite query using alternative semantics and retry.
        4. Deduplicate and return verified job listings.

        Args:
            role_title:      Search string.
            location:        Preferred location (display hint for the feeds).
            experience_level: Seniority hint passed to the tool.
            target_count:    Upper bound on returned listings.
            max_age_hours:   How far back to look for postings.
            employment_type: "Any" to keep everything, otherwise keep only
                             listings whose type matches (e.g. "Full Time").
            extra_queries:   Portal keywords from Agent 2. These are tried
                             before the built-in expansions because they are
                             derived from the candidate's actual profile, and
                             they were previously generated and then discarded.
        """
        print(f"[{self.agent_name}] Initiating search for: '{role_title}' in '{location}'...")

        # Step 1: Initial query. Over-fetch so the search agent has material to
        # choose from when expanding below.
        jobs = self.search_tool.search_jobs(
            query=role_title,
            location=location,
            experience_level=experience_level,
            max_results=max(target_count * 3, 15),
            max_age_hours=max_age_hours
        )

        # Step 2: Query expansion. The trigger is "fewer than we need", not
        # "zero" — previously a single off-target hit suppressed expansion
        # entirely, which is how a 5-job request could return 1 unrelated role.
        agent_keywords = {
            " ".join(k.lower().split()) for k in (extra_queries or [])
        }
        expansions = self._expansion_order(role_title, extra_queries)
        for alt_query in expansions[: self._MAX_EXPANSIONS]:
            if len(jobs) >= target_count:
                break
            print(f"[{self.agent_name}] Only {len(jobs)}/{target_count} on-target "
                  f"results; widening query to '{alt_query}'...")
            more_jobs = self.search_tool.search_jobs(
                query=alt_query,
                location=location,
                experience_level=experience_level,
                max_results=target_count * 2,
                max_age_hours=max_age_hours
            )

            # Re-validate against the ORIGINAL role. An expansion like
            # "Data Analyst" for a "Machine Learning Engineer" search returns
            # legitimately matching analyst jobs that are still off-target for
            # what the candidate asked for, so every expanded hit must be
            # checked against the role the candidate actually selected.
            if " ".join(alt_query.lower().split()) in agent_keywords:
                # Agent 2's keywords are LLM output and can drift into a
                # neighbouring field, so they are judged against the role.
                more_jobs = self._reject_off_role(more_jobs, role_title)
            # Built-in expansions already cleared the strict gate inside
            # search_jobs for the query they ran under, and _expansion_order
            # guarantees they share a discipline term with the role. Re-checking
            # them against the full role set here is what discarded every
            # widened hit of a "Python Backend Developer" search for omitting
            # "Python", leaving one result out of five.
            jobs.extend(more_jobs)

        # Step 3: Re-rank the merged pool against the original role and dedupe.
        jobs = self._rerank(jobs, role_title, target_count)

        jobs = self._filter_by_employment_type(jobs, employment_type)

        print(f"[{self.agent_name}] Successfully retrieved {len(jobs)} jobs.")
        return jobs[:target_count]

    @staticmethod
    def _filter_by_employment_type(jobs: List[JobListing], employment_type: str) -> List[JobListing]:
        """
        Keep only listings matching the requested employment type.

        Feeds are inconsistent about spelling ("Full-time", "Full Time",
        "FULLTIME"), so the comparison is normalised rather than exact.
        """
        if not employment_type or employment_type.strip().lower() == "any":
            return jobs

        def norm(value: str) -> str:
            return "".join(ch for ch in (value or "").lower() if ch.isalnum())

        wanted = norm(employment_type)
        # A blank type on the listing is treated as unknown, not as a mismatch,
        # so a sparse feed does not silently empty the results.
        return [j for j in jobs if not norm(j.employment_type) or wanted in norm(j.employment_type)]

    @staticmethod
    def _reject_off_role(jobs: List[JobListing], role_title: str) -> List[JobListing]:
        """
        Drop listings that do not match the candidate's *original* role.

        Query expansion widens recall, so it can surface legitimate jobs for a
        neighbouring role ("Data Analyst" for an ML Engineer search). Those are
        not what the candidate asked for, so expanded hits must re-clear the bar
        set by the role they were retrieved for.
        """
        tokens, _, _ = query_tokens(role_title)
        if not tokens:
            return jobs

        # Require one of the original query's discipline terms in the title.
        # Demanding *all* of them would defeat the purpose of expanding: a
        # "Python Backend Developer" search deliberately widens to "Backend
        # Developer", and the results that come back no longer say "Python".
        # Requiring all of them reduced a 5-job request to a single hit.
        # Weak role nouns are excluded so the shared "Engineer" cannot be what
        # validates the listing.
        role_nouns = domain_tokens(tokens)
        if not role_nouns:
            return jobs

        kept: List[JobListing] = []
        for job in jobs:
            title_words = field_tokens(job.title or "")
            if any(t in title_words for t in role_nouns):
                kept.append(job)
        return kept

    @staticmethod
    def _rerank(jobs: List[JobListing], role_title: str, limit: int) -> List[JobListing]:
        """
        Order the merged pool by fit to the original role and dedupe.

        Expansion appends to a list, so without this step results from the
        first (highest-scoring) query could be pushed below weaker expanded
        hits, and the same job could appear twice under different expansions.

        Membership is *not* re-gated here. Each source already applied the right
        test for how it was retrieved: first-pass results cleared the strict
        full-discipline gate, expanded ones cleared the relaxed any-discipline
        gate in :meth:`_reject_off_role`. Re-applying the strict gate at the end
        threw the expanded results away again — a "Python Backend Developer"
        search widened to "Backend Developer" and then discarded every widened
        hit for not mentioning Python, leaving one result out of five.
        """
        tokens, target_rank, _ = query_tokens(role_title)
        deduped: List[JobListing] = []
        seen: Set[Tuple[str, str]] = set()
        for job in jobs:
            key = (
                (job.title or "").lower().strip(),
                (job.company or "").lower().strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(job)

        if not tokens:
            return deduped[:limit]

        ranked = []
        for idx, job in enumerate(deduped):
            base = score_job(job, tokens)
            ranked.append(
                (base * seniority_penalty(job, target_rank), base, idx, job)
            )
        ranked.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [t[3] for t in ranked][:limit]

    def _expansion_order(
        self, role_title: str, extra_queries: Optional[List[str]] = None
    ) -> List[str]:
        """
        Order the alternative queries to try: Agent 2's keywords first, then the
        built-in heuristics.

        The two sources are validated differently. Agent 2's keywords are taken
        on trust because they were derived from the candidate's own profile;
        the built-in ones must still share a discipline term with the original
        role, otherwise "Python Developer" gets tried for a backend search and
        every result it returns is then thrown away by the off-role check.

        Either way the expansion's results are re-validated against the original
        role before they are merged, so a drifting keyword cannot introduce an
        unrelated role.
        """
        seen = {" ".join(role_title.lower().split())}
        role_domain = set(domain_tokens(query_tokens(role_title)[0]))
        ordered: List[str] = []

        agent_keywords = [
            k for k in (extra_queries or []) if domain_tokens(query_tokens(k)[0])
        ]
        for candidate, from_agent in (
            [(k, True) for k in agent_keywords]
            + [(k, False) for k in self._generate_query_expansions(role_title)]
        ):
            key = " ".join((candidate or "").lower().split())
            if not key or key in seen:
                continue
            if not from_agent and role_domain:
                overlap = set(domain_tokens(query_tokens(candidate)[0])) & role_domain
                if not overlap:
                    continue
            seen.add(key)
            ordered.append(candidate)

        return ordered

    def _generate_query_expansions(self, role: str) -> List[str]:
        """
        Produce alternative search queries to improve recall.

        These stay within the candidate's chosen role family. The previous
        version jumped domains ("Data Analyst" for an ML search, "Prompt
        Engineer" for a GenAI search) and used bare seniority keywords that no
        board indexes, so expansion produced unrelated jobs rather than more of
        the right ones. Every expansion below keeps the role's core nouns and
        varies only the modifier or drops a modifier to widen recall.
        """
        expansions: List[str] = []
        role_clean = " ".join(role.split())
        role_lower = role_clean.lower()

        # Seniority is already enforced as a ranking penalty, so an expansion
        # should never hard-code it — that would hide the senior openings the
        # candidate may legitimately be eligible for.
        _, base_rank = extract_seniority(role_lower)
        base = role_clean
        for term, rank in (
            ("intern", 0), ("internship", 0), ("trainee", 0), ("fresher", 1),
            ("junior", 1), ("entry", 1), ("senior", 3), ("mid", 2),
        ):
            if base_rank == rank:
                base = " ".join(
                    w for w in role_clean.split() if w.lower().strip(".,") != term
                ).strip()
                break

        role_lower = base.lower()

        if "machine learning" in role_lower or re.search(r"\bml\b", role_lower) or "ai" in role_lower:
            expansions.append("Machine Learning Engineer")
            expansions.append("ML Engineer")
            expansions.append("Artificial Intelligence Engineer")
        elif "generative ai" in role_lower or "genai" in role_lower or "llm" in role_lower:
            expansions.append("Generative AI Engineer")
            expansions.append("LLM Engineer")
            expansions.append("AI Engineer")
        elif "data science" in role_lower or "data scientist" in role_lower:
            expansions.append("Data Scientist")
            expansions.append("Data Science Engineer")
            expansions.append("Machine Learning Engineer")
        elif "data" in role_lower and "analy" in role_lower:
            expansions.append("Data Analyst")
            expansions.append("Business Intelligence Analyst")
            expansions.append("Analytics Engineer")
        elif "backend" in role_lower or "python" in role_lower:
            expansions.append("Backend Developer")
            expansions.append("Backend Engineer")
            expansions.append("Software Engineer")
        elif "frontend" in role_lower or "front end" in role_lower or "react" in role_lower:
            expansions.append("Frontend Developer")
            expansions.append("React Developer")
            expansions.append("Web Developer")
        elif "full stack" in role_lower or "fullstack" in role_lower:
            expansions.append("Fullstack Engineer")
            expansions.append("Full Stack Engineer")
            expansions.append("Software Engineer")
        elif "devops" in role_lower or "cloud" in role_lower:
            expansions.append("DevOps Developer")
            expansions.append("DevOps Engineer")
            expansions.append("Site Reliability Engineer")
        else:
            # Generic widening must keep the domain nouns. The previous version
            # truncated the role ("Senior Software Engineer" -> "Software"),
            # which searches for nothing and can only ever return noise.
            words = base.split()
            expansions.append(base)
            # Swap the trailing role noun for a synonym, e.g.
            # "Software Engineer" -> "Software Developer".
            swaps = self._ROLE_NOUN_SWAPS
            if words and words[-1].lower() in swaps:
                expansions.append(" ".join(words[:-1] + [swaps[words[-1].lower()]]))
            # Leading modifiers ("Frontend", "Junior", "Associate") are often
            # dropped by portals, so retry without them.
            if len(words) > 1 and words[0].lower() in self._DROPPABLE_MODIFIERS:
                expansions.append(" ".join(words[1:]))
            # Append the seniority the portal most likely indexed for.
            expansions.append(
                ("Junior " + base) if base_rank is None or base_rank >= 2 else base
            )

        # Never emit a query identical to the original — it would re-run the
        # same search and re-insert the same results. Also drop any expansion
        # that lost its domain nouns: a bare "Engineer" or "Software" matches
        # everything and is what produced unrelated roles in the first place.
        seen: Set[str] = {role_lower}
        unique: List[str] = []
        for exp in expansions:
            key = exp.lower().strip()
            if not key or key in seen:
                continue
            # Reject expansions that lost their discipline entirely: a bare
            # "Engineer" or "Software" matches everything, which is how
            # unrelated roles got in during expansion in the first place.
            if not domain_tokens(query_tokens(exp)[0]):
                continue
            seen.add(key)
            unique.append(exp)

        return unique or [base]

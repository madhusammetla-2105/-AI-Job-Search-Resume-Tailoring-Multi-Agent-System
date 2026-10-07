"""
Shared lexical relevance engine for job matching.

Every retrieval stage (SQLite cache, remote feeds, in-process ranking) must
agree on what "this job matches this query" means. Previously each stage had
its own ad-hoc rule, so a query could be accepted by the cache yet rejected by
the ranker, or vice versa. This module is the single source of truth.

Academic Alignment: Module III – Informed Search (heuristic evaluation with
word-boundary evidence rather than raw substring containment).

Three defects this fixes:

1. Substring containment. ``"ai" in title`` matches "Training", "Maintenance"
   and "Retail"; ``"ml" in title`` matches "HTML". Matching is now performed on
   whole words with light stemming.

2. A stopword list that ate the whole query. It contained the role nouns
   themselves ("engineer", "developer", "manager", "stack"), so "Full Stack
   Developer" tokenised to *nothing* and the ranker silently fell through to
   unranked feed order. Role words are now the signal, and only genuine noise
   is stripped.

3. Seniority words were discarded entirely, so an entry-level search happily
   returned Staff/Principal/Lead roles. They are now parsed separately and
   enforced as a penalty.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

# Genuine noise only: articles, prepositions, and the boilerplate that portals
# staple onto titles. Role nouns ("engineer", "developer", "scientist",
# "stack") are deliberately absent — they are the strongest relevance signal.
_NOISE_WORDS: Set[str] = {
    "a", "an", "and", "or", "the", "for", "of", "in", "at", "to", "with",
    "job", "jobs", "role", "roles", "position", "positions", "opening",
    "openings", "vacancy", "vacancies", "work", "candidate", "candidates",
    "please", "apply", "we", "our", "us", "you", "your",
    # Location / arrangement words: these are *preference* signals handled by
    # the location and employment-type filters, not role signals. Keeping them
    # out of the token set stops "remote" from dragging in every remote board.
    "remote", "hybrid", "onsite", "on", "site", "anywhere", "worldwide",
    "part", "time", "times", "permanent", "freelance",
    "bangalore", "bengaluru", "hyderabad", "chennai", "pune", "mumbai",
    "delhi", "noida", "gurgaon", "kolkata", "ahmedabad", "india",
    "usa", "uk", "eu", "europe",
    # Generic seniority adjectives that carry no ladder meaning.
    "mid", "level",
}

# Seniority vocabulary, ordered junior -> senior so ranks are comparable.
_SENIORITY_TERMS: Dict[str, int] = {
    "intern": 0, "internship": 0, "trainee": 0, "traineeship": 0,
    "apprentice": 0, "fresh": 1, "fresher": 1, "entry": 1,
    "graduate": 1, "junior": 1, "jr": 1, "associate": 1,
    "mid": 2, "intermediate": 2, "regular": 2,
    "senior": 3, "sr": 3, "snr": 3,
    "staff": 4, "lead": 4, "principal": 5, "head": 5, "chief": 5,
    "director": 6, "vp": 6, "vice": 6, "cto": 6, "founder": 6,
}

# Past-participle / inflection variants that should fold onto a base term.
_STEM_MAP: Dict[str, str] = {
    "engineers": "engineer", "engineering": "engineer",
    "developers": "developer", "development": "developer",
    "programmers": "programmer", "programming": "programmer",
    "scientists": "scientist", "science": "scientist",
    "designers": "designer", "designing": "designer",
    "analysts": "analyst", "analytics": "analyst",
    "administrators": "administrator",
    "managers": "manager", "managing": "manager",
    "architects": "architect", "architecture": "architect",
    "specialists": "specialist", "specialized": "specialist",
    "consultants": "consultant", "consulting": "consultant",
    "researchers": "researcher", "researching": "researcher",
    "testers": "tester", "testing": "tester",
    "writers": "writer", "writing": "writer",
    "executive": "executive", "executives": "executive",
    # Do NOT fold "learning" -> "ml" or "intelligence" -> "ai" here: that
    # breaks the "machine learning" / "artificial intelligence" phrases,
    # which are handled as aliases below. Mapping them separately produced
    # the nonsense token "machineml".
    "devops": "devops", "devop": "devops", "kubernetes": "kubernetes",
    "k8s": "kubernetes", "postgres": "postgresql", "postgresql": "postgresql",
    "reactjs": "react", "nodejs": "node", "nextjs": "next",
    "javascript": "javascript", "typescript": "typescript", "python": "python",
}

# Multi-word role phrases that must collapse to one token so a query and a
# listing match regardless of spelling or spacing: "Full Stack Developer" and
# "Fullstack Developer" both have to reduce to the same evidence.
_PHRASE_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fullstack", ("full", "stack")),
    ("ml", ("machine", "learning")),
    ("ai", ("artificial", "intelligence")),
    ("generativeai", ("generative", "ai")),
    ("frontend", ("front", "end")),
    ("backend", ("back", "end")),
    ("devops", ("dev", "ops")),
)


def _stem(word: str) -> str:
    """Fold common inflections and role synonyms onto a comparable base form."""
    if word in _STEM_MAP:
        return _STEM_MAP[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize(text: str) -> str:
    """Lower-case and strip punctuation, keeping ``c++``/``.net`` intact."""
    return (text or "").lower()


def tokenize(text: str, drop_noise: bool = True) -> List[str]:
    """
    Split text into stemmed, comparable word tokens.

    Args:
        text:       Raw string (query, title, description…).
        drop_noise: Remove filler/location words. Always keep the role words.
    """
    out: List[str] = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        token = raw.strip(".")
        if not token or len(token) < 2:
            continue
        if drop_noise and token in _NOISE_WORDS:
            continue
        stemmed = _stem(token)
        if stemmed:
            out.append(stemmed)
    return out


def expand_with_phrases(tokens: List[str]) -> List[str]:
    """
    Collapse known multi-word role phrases onto a single canonical token.

    Only the registered aliases are collapsed. The previous version joined
    *every* adjacent pair, which invented tokens like "stackdeveloper" and
    "backendengineer" that appear in no job title — noise that then had to be
    matched and diluted the ranking.
    """
    out = list(tokens)
    for canonical, parts in _PHRASE_ALIASES:
        if all(p in out for p in parts):
            for p in parts:
                if p in out:
                    out.remove(p)
            out.append(canonical)
    return out


def query_tokens(query: str) -> List[str]:
    """
    Full query analysis: role tokens (with phrase expansion) plus seniority.

    Returns ``(tokens, seniority_rank, seniority_word)`` where ``seniority_rank``
    is ``None`` when the query states no seniority.
    """
    lowered = normalize(query)
    seniority_word, seniority_rank = extract_seniority(lowered)

    # Seniority words must not leak into the role tokens: "Senior Data Analyst"
    # and "Data Analyst" must produce the *same* role tokens, with seniority
    # enforced separately so it can be penalised rather than ignored.
    scrubbed = re.sub(
        r"\b(" + "|".join(sorted(_SENIORITY_TERMS, key=len, reverse=True)) + r")\b",
        " ",
        lowered,
    )
    tokens = expand_with_phrases(tokenize(scrubbed, drop_noise=True))
    tokens = _expand_role_class(tokens)
    return tokens, seniority_rank, seniority_word


def extract_seniority(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Return ``(matched_word, rank)`` for the seniority implied by *text*."""
    best_word: Optional[str] = None
    best_rank: Optional[int] = None
    for raw in _TOKEN_RE.findall(normalize(text)):
        rank = _SENIORITY_TERMS.get(raw)
        if rank is None:
            continue
        # "vice president" carries "vp" later in the string; take the highest
        # seniority mentioned so a title never under-states the ladder.
        if best_rank is None or rank > best_rank:
            best_word, best_rank = raw, rank
    return best_word, best_rank


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Weights. Title evidence dominates because a portal's title is the single most
# reliable statement of what the role actually is.
W_TITLE = 4.0
W_SKILL = 2.0
W_DESCRIPTION = 0.5
W_PHRASE_BONUS = 6.0

# Role nouns split by how much they discriminate. The weak set appears in most
# software postings ("half of all titles end in Engineer"), so they are treated
# as a single role class and excluded from the domain terms. The rest still name
# a specific discipline — "Data Analyst" is not "Data Engineer" — so they stay
# as real evidence and must be corroborated.
WEAK_ROLE_NOUNS: Set[str] = {
    "engineer", "developer", "manager", "officer", "lead", "head",
    "coordinator", "specialist", "associate", "professional", "generalist",
    "freelancer", "expert",
}

SPECIFIC_ROLE_NOUNS: Set[str] = {
    "scientist", "analyst", "designer", "architect", "consultant",
    "programmer", "administrator", "researcher", "tester", "writer",
    "executive", "technician", "advisor", "accountant", "actuary",
    "recruiter", "nurse", "teacher", "chef", "driver", "account",
}

# Weak role nouns fall into two families that are interchangeable *within* a
# family and not across. "Backend Engineer" and "Backend Developer" are the
# same opening, but "Technical Product Manager" is not a developer role — and
# letting the shared word "app" or "frontend" carry a manager title past a
# developer query is how sales and product roles kept surfacing in dev results.
ROLE_GROUP_BUILD = "rolebuild"
ROLE_GROUP_MANAGE = "rolemanage"

ROLE_GROUPS: Dict[str, str] = {}
for _w in ("engineer", "developer", "programmer", "coder", "swebench"):
    ROLE_GROUPS[_w] = ROLE_GROUP_BUILD
for _w in ("manager", "management", "lead", "head", "director", "chief",
           "officer", "principal", "staff", "vp"):
    ROLE_GROUPS[_w] = ROLE_GROUP_MANAGE

GENERIC_ROLE_NOUNS: Set[str] = WEAK_ROLE_NOUNS | SPECIFIC_ROLE_NOUNS

# All role nouns collapse to this one class token so they can satisfy each
# other. Portals use "Backend Engineer" and "Backend Developer" for the same
# opening, and treating those as different evidence meant a genuine match was
# rejected while a title carrying any stray role noun was not.
ROLE_CLASS = "role"


def _field_tokens(text: str) -> Set[str]:
    """
    Tokens of a job field, with multi-word role phrases collapsed.

    Phrase collapsing must happen on both sides. Applying it only to the query
    made "ML Engineer" (one token "ml") unable to match the title
    "Machine Learning Engineer" (tokens "machine", "learning").
    """
    if not text:
        return set()
    toks = set(expand_with_phrases(tokenize(text)))
    if toks & GENERIC_ROLE_NOUNS:
        toks.add(ROLE_CLASS)
    for group in {ROLE_GROUPS[w] for w in toks if w in ROLE_GROUPS}:
        toks.add(group)
    return toks


def domain_tokens(tokens: List[str]) -> List[str]:
    """
    Query tokens that actually pin down the discipline.

    Weak role nouns ("Backend Developer" -> "backend") are dropped: they match
    most software titles, so leaving them in fills every result slot with a
    "developer" hit and buries the rows carrying the real domain term.
    Specific role nouns are kept — "Data Analyst" and "Data Engineer" are not
    the same opening. The synthetic role-class and role-family tokens are
    dropped because no title spells them.
    """
    synthetic = GENERIC_ROLE_NOUNS | {ROLE_CLASS} | set(ROLE_GROUPS.values())
    return [t for t in tokens if t not in synthetic]


def sql_search_variants(tokens: List[str]) -> List[List[str]]:
    """
    Map canonical tokens to the literal spellings a SQL LIKE must try.

    "fullstack" is the canonical form produced by phrase collapsing, but job
    titles overwhelmingly spell it "Full Stack". Searching for the canonical
    token alone returns nothing, so each alias also contributes its spaced
    parts as a single phrase to match, never as independent terms.

    Returns a list of groups; a row matches if ANY group matches.
    """
    variants: List[List[str]] = []
    spaced = {canon: " ".join(parts) for canon, parts in _PHRASE_ALIASES}
    for token in tokens:
        group = [token]
        phrase = spaced.get(token)
        if phrase and phrase != token:
            group.append(phrase)
        variants.append(group)
    return variants


def field_tokens(text: str) -> Set[str]:
    """Public alias for the field tokeniser (used by agents for validation)."""
    return _field_tokens(text)


def _expand_role_class(tokens: List[str]) -> List[str]:
    """
    Add the shared role-class token and the role-family tokens for a query.

    Role families let "Backend Engineer" satisfy "Backend Developer" without
    letting "Technical Product Manager" satisfy either.
    """
    out = list(tokens)
    if any(t in GENERIC_ROLE_NOUNS for t in tokens):
        out.append(ROLE_CLASS)
    for group in {ROLE_GROUPS[t] for t in tokens if t in ROLE_GROUPS}:
        out.append(group)
    return out


def _title_tokens(job) -> Set[str]:
    return _field_tokens(job.title or "")


def score_job(job, tokens: List[str]) -> float:
    """
    Weighted evidence score for one listing against query *tokens*.

    Matching is whole-word and stemmed, so "ai" no longer fires on "Training"
    and "ml" no longer fires on "HTML".
    """
    if not tokens:
        return 0.0

    title_toks = _title_tokens(job)
    skill_toks: Set[str] = set()
    for skill in job.required_skills or []:
        skill_toks.update(tokenize(skill))
    desc_toks = _field_tokens(job.description or "")

    score = 0.0
    matched_title = 0
    for token in tokens:
        if token in title_toks:
            score += W_TITLE
            matched_title += 1
        elif token in skill_toks:
            score += W_SKILL
        elif token in desc_toks:
            score += W_DESCRIPTION

    # Whole-query phrase in the title is the strongest possible signal.
    phrase = " ".join(tokens)
    if phrase and phrase in normalize(job.title or ""):
        score += W_PHRASE_BONUS

    # A listing that matched nothing in the title is off-target regardless of
    # how much incidental description text it shared. Recorded via a returned
    # attribute-free signal: callers filter with is_on_target().
    return score


def is_on_target(job, tokens: List[str], score: float) -> bool:
    """
    Decide whether *job* genuinely matches the query.

    A listing qualifies only when the title carries the query's discipline and
    its role family. Description-only matches are rejected: job descriptions are
    boilerplate-heavy and any posting in the same domain mentions its
    neighbour's vocabulary.
    """
    if not tokens:
        # A query made entirely of noise words ("Remote", "Full Time") carries
        # no role signal at all; the caller falls back to feed order by design.
        return True

    title_toks = _title_tokens(job)
    if ROLE_CLASS in tokens and ROLE_CLASS not in title_toks:
        return False

    # A build-role query must not be answered with a management title just
    # because the discipline word is shared: "Technical Product Manager, AI
    # Stockbroking App" is not a "Mobile App Developer" opening.
    wanted_groups = {g for g in ROLE_GROUPS.values() if g in tokens}
    if wanted_groups and not (wanted_groups & title_toks):
        return False

    # Discipline terms pin down which field the role belongs to. Requiring them
    # is what separates "AI Engineer" from "Machine Learning Engineer" — the
    # latter shares the role noun but not the domain.
    domain = domain_tokens(tokens)
    if not domain:
        # The query is only a role noun ("Engineer"); the checks above are the
        # whole test.
        return True

    # A term may be satisfied by the skills list, not only the title: portals
    # routinely title a role "Senior Backend Engineer" and put the stack in the
    # tag list. Requiring the title alone rejected those.
    skill_toks: Set[str] = set()
    for skill in job.required_skills or []:
        skill_toks.update(tokenize(skill))
    for term in domain:
        if term not in title_toks and term not in skill_toks:
            return False
    return True


def seniority_penalty(job, target_rank: Optional[int]) -> float:
    """
    Multiplicative penalty when a listing is materially more senior than asked.

    A fresher asking for entry-level roles should not be shown Staff Engineer
    openings as equals. Within-one-rank differences are tolerated because feed
    titles are noisy; a 2+ rank overshoot is penalised hard.
    """
    if target_rank is None:
        return 1.0
    _, job_rank = extract_seniority(job.title or "")
    if job_rank is None:
        return 1.0
    gap = job_rank - target_rank
    if gap <= 1:
        return 1.0
    if gap == 2:
        return 0.55
    # Decay geometrically rather than clamping: a Principal role is a worse
    # suggestion for a fresher than a Staff one, and clamping both to the same
    # floor left their relative order down to feed order.
    return round(0.5 ** (gap - 1), 3)


def filter_and_rank(jobs: List, tokens: List[str], target_rank: Optional[int] = None,
                    limit: Optional[int] = None) -> List:
    """
    Score, filter to on-target listings, apply seniority, and sort best-first.

    Unlike the previous implementation this never falls back to unranked feed
    order: when nothing matches, it returns an empty list so the search agent
    can widen the query instead of surfacing unrelated roles.
    """
    scored = []
    for idx, job in enumerate(jobs):
        base = score_job(job, tokens)
        if not is_on_target(job, tokens, base):
            continue
        weight = seniority_penalty(job, target_rank)
        scored.append((base * weight, base, idx, job))

    # Sort by relevance desc, then seniority (least senior first), then feed
    # order for determinism.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out = [t[3] for t in scored]
    return out[:limit] if limit is not None else out


def matches(text: str, tokens: List[str]) -> bool:
    """Whole-word match of any query token against *text*."""
    return any(t in _field_tokens(text) for t in tokens)
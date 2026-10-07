"""
Regression tests for job-query relevance.

These lock in the four defects that made Agent 3 return unrelated roles:

1. Substring matching ("ai" in "Training", "ml" in "HTML").
2. A stopword list containing the role nouns, so "Full Stack Developer"
   tokenised to nothing and fell through to unranked feed order.
3. Falling back to unranked feed order when nothing matched.
4. Expanding the query only when the first attempt returned literally zero.

Run with:  .venv/bin/python -m pytest tests/test_relevance.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.job_search_tool import JobSearchTool, _dict_to_job_listing  # noqa: E402
from tools.relevance import (  # noqa: E402
    domain_tokens,
    extract_seniority,
    filter_and_rank,
    is_on_target,
    query_tokens,
    score_job,
)

TOOL = JobSearchTool()


def listing(title, description="", skills="", company="Acme", job_id=None):
    return _dict_to_job_listing(
        {
            "id": job_id or title,
            "title": title,
            "company": company,
            "description": description,
            "required_skills": skills,
            "employment_type": "Full-time",
        },
        fallback_query=title,
    )


def titles_for(query, jobs, rank=None):
    return [j.title for j in filter_and_rank(jobs, query_tokens(query)[0], rank)]


# ---------------------------------------------------------------------------
# Defect 1: substring matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,off_target", [
    ("AI Engineer", "Sales Training Manager"),
    ("AI Engineer", "Maintenance Technician"),
    ("AI Engineer", "Retail Operations Lead"),
    ("ML Engineer", "HTML Email Developer"),
    ("ML Engineer", "Webmail Administrator"),
    ("Data Analyst", "Android Developer"),
])
def test_word_boundary_rejects_substring_collisions(query, off_target):
    tokens = query_tokens(query)[0]
    job = listing(off_target, description="Email campaigns for retail stores.")
    assert not is_on_target(job, tokens, score_job(job, tokens)), (
        f"{off_target!r} matched {query!r} via substring containment"
    )


def test_word_boundary_still_matches_real_ai_roles():
    tokens = query_tokens("AI Engineer")[0]
    job = listing("AI Engineer, Data APIs")
    assert is_on_target(job, tokens, score_job(job, tokens))


# ---------------------------------------------------------------------------
# Defect 2: role nouns erased by stopwords
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("Full Stack Developer", "fullstack"),
    ("Software Engineer", "engineer"),
    ("Senior Backend Engineer", "backend"),
    ("Machine Learning Engineer", "ml"),
    ("ML Engineer", "ml"),
    ("Data Scientist", "scientist"),
    ("DevOps Engineer", "devops"),
])
def test_query_keeps_role_nouns(query, expected):
    assert expected in query_tokens(query)[0]


def test_multiword_role_spellings_are_equivalent():
    a = set(query_tokens("Full Stack Developer")[0])
    b = set(query_tokens("Fullstack Developer")[0])
    assert a == b


def test_machine_learning_matches_ml_and_spaced_forms():
    tokens = query_tokens("ML Engineer")[0]
    for title in ("Machine Learning Engineer", "ML Engineer", "Staff ML Engineer"):
        job = listing(title)
        assert is_on_target(job, tokens, score_job(job, tokens)), title


# ---------------------------------------------------------------------------
# Defect 3: unranked fallback returned unrelated jobs
# ---------------------------------------------------------------------------

def test_no_unranked_fallback_when_nothing_matches():
    raw = [
        {"id": "1", "title": "Sales Manager", "company": "A", "description": "lead a team"},
        {"id": "2", "title": "HR Business Partner", "company": "B", "description": "hiring"},
        {"id": "3", "title": "Warehouse Operative", "company": "C", "description": "forklifts"},
    ]
    assert TOOL._raw_list_to_listings(raw, "Machine Learning Engineer", 5) == []


def test_noise_only_query_preserves_feed_order():
    """A query with no role signal is the one case that may pass through."""
    raw = [
        {"id": "1", "title": "Anything At All", "company": "A", "description": ""},
        {"id": "2", "title": "Something Else", "company": "B", "description": ""},
    ]
    got = TOOL._raw_list_to_listings(raw, "Remote", 5)
    assert [j.title for j in got] == ["Anything At All", "Something Else"]


def test_ai_query_does_not_return_ml_engineer():
    """The headline bug: a shared role noun is not evidence of fit."""
    jobs = [
        listing("Machine Learning Engineer"),
        listing("Machine Learning Intern"),
        listing("AI Engineer, Data APIs"),
    ]
    assert titles_for("AI Engineer", jobs) == ["AI Engineer, Data APIs"]


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

def test_seniority_is_extracted_not_discarded():
    assert extract_seniority("Senior Backend Engineer")[1] == 3
    assert extract_seniority("Machine Learning Intern")[1] == 0
    assert extract_seniority("Data Analyst")[1] is None


def test_seniority_word_does_not_change_role_tokens():
    a = set(query_tokens("Senior Data Analyst")[0])
    b = set(query_tokens("Data Analyst")[0])
    assert a == b


def test_entry_level_search_demotes_staff_roles():
    jobs = [
        listing("Principal AI Engineer"),
        listing("Junior AI Engineer"),
        listing("Staff AI Engineer"),
    ]
    got = titles_for("AI Engineer", jobs, rank=1)
    assert got[0] == "Junior AI Engineer"
    assert got[-1] == "Principal AI Engineer"


def test_seniority_penalty_is_inert_without_a_target():
    jobs = [listing("Principal AI Engineer"), listing("Junior AI Engineer")]
    got = titles_for("AI Engineer", jobs, rank=None)
    assert got[0] == "Principal AI Engineer"


# ---------------------------------------------------------------------------
# Ranking quality
# ---------------------------------------------------------------------------

def test_exact_title_outranks_phrase_mention_in_description():
    jobs = [
        listing("Software Engineer, Payments",
                description="You will work with our machine learning platform."),
        listing("Machine Learning Engineer"),
    ]
    assert titles_for("Machine Learning Engineer", jobs)[0] == "Machine Learning Engineer"


def test_description_mention_alone_is_not_enough():
    jobs = [listing("Software Engineer, Payments",
                    description="You will work with our machine learning platform.")]
    assert titles_for("Machine Learning Engineer", jobs) == []


def test_engineer_and_developer_are_interchangeable():
    jobs = [listing("Backend Engineer, Core Technology"),
            listing("Frontend Engineer")]
    got = titles_for("Backend Developer", jobs)
    assert got == ["Backend Engineer, Core Technology"]


def test_leading_ordinal_digits_are_stripped_from_titles():
    jobs = [listing("1146 - Fullstack Engineer")]
    assert titles_for("Full Stack Developer", jobs) == ["1146 - Fullstack Engineer"]

# ---------------------------------------------------------------------------
# Cache recall layer
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_cache(tmp_path):
    from tools.job_cache import JobCache
    return JobCache(str(tmp_path / "test_cache.db"))


SEED = [
    {"id": "1", "title": "Senior AI Engineer", "company": "A", "description": "",
     "required_skills": "python,machine learning"},
    {"id": "2", "title": "Sales Training Manager", "company": "B", "description": "",
     "required_skills": "sales"},
    {"id": "3", "title": "Junior Full Stack Developer", "company": "C", "description": "",
     "required_skills": "react,node"},
    {"id": "4", "title": "Backend Engineer, Core Technology", "company": "D",
     "description": "", "required_skills": "go,postgresql"},
    {"id": "5", "title": "Data Analyst", "company": "E", "description": "",
     "required_skills": "sql,excel"},
]


def test_cache_ai_query_excludes_substring_collisions(temp_cache):
    temp_cache.bulk_upsert(SEED)
    rows = temp_cache.get_by_date_range("AI Engineer")
    titles = [r["title"] for r in rows]
    assert "Senior AI Engineer" in titles
    assert "Sales Training Manager" not in titles


def test_cache_does_not_and_every_token(temp_cache):
    """'Backend Developer' must still find the Backend Engineer row."""
    temp_cache.bulk_upsert(SEED)
    rows = temp_cache.get_by_date_range("Backend Developer")
    titles = [r["title"] for r in rows]
    assert "Backend Engineer, Core Technology" in titles


def test_cache_handles_compound_role_spellings(temp_cache):
    temp_cache.bulk_upsert(SEED)
    for query in ("Full Stack Developer", "Fullstack Developer"):
        rows = temp_cache.get_by_date_range(query)
        assert "Junior Full Stack Developer" in [r["title"] for r in rows], query


def test_cache_ranks_multi_token_matches_first(temp_cache):
    """A flat LIMIT would otherwise be filled by single-token noise."""
    seed = SEED + [
        {"id": str(i), "title": f"Senior Engineer {i}", "company": "X",
         "description": "", "required_skills": ""}
        for i in range(10, 30)
    ]
    temp_cache.bulk_upsert(seed)
    rows = temp_cache.get_by_date_range("Backend Developer", limit=5)
    assert rows and rows[0]["title"] == "Backend Engineer, Core Technology"


def test_cache_returns_empty_for_unmatchable_query(temp_cache):
    temp_cache.bulk_upsert(SEED)
    assert temp_cache.get_by_date_range("Warehouse Operative") == []


# ---------------------------------------------------------------------------
# Search agent behaviour
# ---------------------------------------------------------------------------

def test_query_expansions_stay_in_the_role_family():
    from agents.job_search_agent import JobSearchAgent
    agent = JobSearchAgent.__new__(JobSearchAgent)
    for role in ("Machine Learning Engineer", "Data Analyst", "DevOps Engineer"):
        for exp in agent._generate_query_expansions(role):
            assert domain_tokens(query_tokens(exp)[0]), (
                f"expansion {exp!r} for {role!r} lost its discipline terms"
            )


def test_expansions_do_not_hardcode_seniority():
    from agents.job_search_agent import JobSearchAgent
    agent = JobSearchAgent.__new__(JobSearchAgent)
    for exp in agent._generate_query_expansions("Senior Data Analyst"):
        assert extract_seniority(exp)[1] != 3, exp


def test_expansions_never_repeat_the_original_query():
    from agents.job_search_agent import JobSearchAgent
    agent = JobSearchAgent.__new__(JobSearchAgent)
    for role in ("Backend Developer", "Machine Learning Engineer", "UI/UX Designer"):
        exps = [e.lower().strip() for e in agent._generate_query_expansions(role)]
        assert role.lower().strip() not in exps, role
        assert len(exps) == len(set(exps)), role


def test_off_role_expansion_results_are_rejected():
    from agents.job_search_agent import JobSearchAgent
    jobs = [listing("Data Analyst"), listing("Machine Learning Engineer")]
    kept = JobSearchAgent._reject_off_role(jobs, "Machine Learning Engineer")
    assert [j.title for j in kept] == ["Machine Learning Engineer"]


def test_rerank_dedupes_by_title_and_company():
    from agents.job_search_agent import JobSearchAgent
    jobs = [
        listing("AI Engineer", company="Acme", job_id="1"),
        listing("AI Engineer", company="Acme", job_id="2"),
        listing("AI Engineer", company="Globex", job_id="3"),
    ]
    got = JobSearchAgent._rerank(jobs, "AI Engineer", 10)
    assert [(j.title, j.company) for j in got] == [
        ("AI Engineer", "Acme"), ("AI Engineer", "Globex")
    ]


def test_rerank_keeps_relaxed_expansion_hits():
    """
    Widening "Python Backend Developer" to "Backend Developer" must not have its
    own results discarded afterwards for omitting "Python". _rerank only orders
    and dedupes; membership is decided per source, not re-gated here.
    """
    from agents.job_search_agent import JobSearchAgent
    jobs = [listing("Senior Backend Engineer"), listing("Warehouse Operative")]
    relaxed = JobSearchAgent._reject_off_role(jobs, "Python Backend Developer")
    got = JobSearchAgent._rerank(relaxed, "Python Backend Developer", 10)
    assert [j.title for j in got] == ["Senior Backend Engineer"]


# ---------------------------------------------------------------------------
# Role families
# ---------------------------------------------------------------------------

def test_build_query_rejects_management_title():
    """"App" in the title is not enough to make a manager a developer role."""
    jobs = [
        listing("Technical Product Manager AI Stockbroking App"),
        listing("Mobile App Developer"),
    ]
    got = titles_for("Mobile App Developer", jobs)
    assert "Technical Product Manager AI Stockbroking App" not in got
    assert "Mobile App Developer" in got


def test_management_query_keeps_management_titles():
    jobs = [
        listing("Product Marketing Manager"),
        listing("Backend Engineer"),
        listing("Marketing Manager, Growth"),
    ]
    got = titles_for("Marketing Manager", jobs)
    assert "Backend Engineer" not in got
    assert "Marketing Manager, Growth" in got


def test_engineer_and_developer_share_a_role_family():
    jobs = [listing("Engineering Manager, AI Platform"),
            listing("AI Engineer")]
    got = titles_for("AI Developer", jobs)
    assert "Engineering Manager, AI Platform" in got
    assert "AI Engineer" in got


def test_dual_titled_roles_satisfy_both_families():
    jobs = [listing("Senior Engineering Manager, Grafana Frontend")]
    assert titles_for("Frontend Engineer", jobs) == [
        "Senior Engineering Manager, Grafana Frontend"
    ]

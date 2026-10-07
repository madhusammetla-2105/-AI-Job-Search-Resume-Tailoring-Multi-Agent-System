# 🎯 Autonomous Multi-Agent AI Job Search & Tailored Resume Suite

An end-to-end, multi-agent AI system designed for students, freshers, and job seekers. The system parses resumes into an ontological knowledge representation, formulates career goals, autonomously discovers matching jobs across web portals, evaluates ATS compatibility with deductive gap analysis, and orchestrates an anti-hallucination multi-agent loop to generate and compile job-specific ATS-compliant PDF resumes.

---

## 🌟 Key Features & Agent Architecture

1. **Agent 1: Resume Parser & Knowledge Extractor (`ResumeParserAgent`)**
   * **Course Alignment:** Module VI (Knowledge Representation & Ontological Engineering).
   * Parses unstructured candidate PDF resumes into a strictly typed `CandidateProfile` ontology.
2. **Agent 2: Career Strategy & Role Recommender (`RoleRecommenderAgent`)**
   * **Course Alignment:** Module VII (State Space Planning & Goal Formulation).
   * Identifies candidate strengths and formulates target goal states (top 3–5 high-fit roles with rationale).
3. **Agent 3: Job Search & Query Optimizer Agent (`JobSearchAgent`)**
   * **Course Alignment:** Modules II & III (Uninformed & Informed Search, Query Expansion).
   * Dynamically rewrites and expands search terms when search results are sparse, retrieving live openings.
4. **Agent 4: Match & Gap Analyzer Agent (`MatchGapAgent`)**
   * **Course Alignment:** Module III (Heuristic Evaluation) & Module V (Inference & Logic).
   * Computes multidimensional ATS match scores and categorizes matched vs missing mandatory skills.
5. **Agent 5: Resume Tailor Agent (`TailorAgent`)**
   * **Course Alignment:** Module VII (State Space Content Synthesis).
   * Aligns summaries, prioritizes keywords, and enhances project bullets truthfully using JD action verbs.
6. **Agent 6: Verifier & Anti-Hallucination Auditor (`VerifierAgent`)**
   * **Course Alignment:** Module IV (Constraint Satisfaction - CSP) & Module IX (Multi-Agent Systems).
   * Acts as an adversarial critic enforcing the grounding constraint: $\text{Tailored Resume} \subseteq \text{Candidate Truth}$.
7. **Document Engine: ATS PDF Compiler (`ReportLab`)**
   * Compiles validated JSON into single-column, ATS-parseable vector PDFs.

---

## 📂 Project Structure

```text
FAI_job_search_multi_task_agent/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration & API settings manager
│
├── schemas/
│   ├── __init__.py
│   ├── resume_schema.py         # CandidateProfile & Contact models
│   ├── job_schema.py            # JobListing & RoleRecommendation models
│   ├── analysis_schema.py       # MatchReport & MatchBreakdown models
│   └── tailoring_schema.py      # TailoredResume & VerificationReport models
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Unified LLM caller with simulation fallback
│   ├── resume_parser_agent.py   # Agent 1: Ontological Parser
│   ├── role_recommender_agent.py# Agent 2: Goal State Formulator
│   ├── job_search_agent.py      # Agent 3: Search & Query Optimizer
│   ├── match_gap_agent.py       # Agent 4: Heuristic ATS & Gap Scorer
│   ├── tailor_agent.py          # Agent 5: Resume Tailor
│   ├── verifier_agent.py        # Agent 6: Grounding Auditor (CSP)
│   └── tailoring_orchestrator.py# Reflection Loop Orchestrator
│
├── tools/
│   ├── __init__.py
│   ├── pdf_reader.py            # PDF text extraction utility
│   ├── job_search_tool.py       # Feed fan-out + ranking against a query
│   ├── remote_feeds.py          # RemoteOK / ArbeitNow / Remotive / Jobicy / LinkedIn
│   ├── job_cache.py             # SQLite cache (recall layer)
│   ├── relevance.py             # Shared word-boundary relevance engine
│   └── pdf_generator.py         # ReportLab ATS PDF generation engine
│
├── ui/
│   ├── __init__.py
│   └── app.py                   # Streamlit Interactive Web Application
│
├── tests/
│   └── test_relevance.py        # Job-query relevance regression suite
│
├── .env.example                 # Configuration template
├── requirements.txt             # Project dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

Create a virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your **Google Gemini API Key** (available free at [Google AI Studio](https://aistudio.google.com/)).
Groq and OpenRouter keys are used as automatic failover providers when present.

### 3. Run the Tests

Validate the job-query relevance rules (no API keys needed):

```bash
pytest tests/ -q
```

### 4. Launch the Streamlit Interactive Dashboard

```bash
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎓 Faculty Presentation Guide (Academic Mapping)

When demonstrating this project to your faculty, highlight how each component implements core syllabus concepts:

| Syllabus Module | Implemented Concept | Where to Show in Code & UI |
| :--- | :--- | :--- |
| **Module II & III: Search** | **Query Expansion & Heuristic Ranking** | `tools/relevance.py` scores listings by whole-word title evidence (no substring matching); `JobSearchAgent` reformulates the query when it returns fewer than the requested number, then re-validates expanded hits against the original role. |
| **Module IV: CSP** | **Constraint Satisfaction on Truthfulness** | `VerifierAgent` checks the constraint: $\text{Claimed Skills} \subseteq \text{Verified Profile Truth}$. |
| **Module V: Inferences** | **Deductive Gap Analysis** | Logical deduction of missing skills and contextual recommendations. |
| **Module VI: Knowledge Rep** | **Ontological Resume Representation** | `CandidateProfile` Pydantic model structuring raw unstructured PDF text. |
| **Module VII: Planning** | **State Space Goal Formulation** | `RoleRecommenderAgent` maps candidate capabilities into 3–5 optimal career goals. |
| **Module IX: Multi-Agent** | **Producer-Critic Reflection Loop** | `TailoringOrchestrator` runs the `TailorAgent` $\leftrightarrow$ `VerifierAgent` feedback convergence loop. |

"""
Streamlit Web Dashboard - AI Job Search and Resume Tailoring Multi-Agent System.

Pipeline: Resume Parsing -> Role Strategy -> Job Discovery -> Gap Analysis -> Tailoring -> PDF.

Design:
  - Grayscale design system (neutral tokens only) with a Light/Dark toggle.
  - Fixed top navigation bar (app identity, theme switch, provider status).
  - Fixed agent workflow bar directly beneath it, listing all six agents.
  - 3:7 horizontal split: left = candidate details + search history,
    right = all pipeline stages.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import html as _html
import re
from contextlib import contextmanager

import streamlit as st
from tools.pdf_reader import extract_text_from_pdf
from tools.pdf_generator import compile_tailored_pdf
from tools.job_search_tool import JobSearchTool
from agents.resume_parser_agent import ResumeParserAgent
from agents.role_recommender_agent import RoleRecommenderAgent
from agents.job_search_agent import JobSearchAgent
from agents.match_gap_agent import MatchGapAgent
from agents.tailoring_orchestrator import TailoringOrchestrator
from config.settings import settings

st.set_page_config(
    page_title="AI Job Search and Resume Tailoring Multi-Agent System",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────────────────────
# Neutral grayscale only. The single chromatic exception is the ATS score
# semantic palette, which must stay distinguishable in a monochrome UI.

THEMES = {
    "light": {
        "bg": "#FFFFFF",
        "surface": "#FAFAFA",
        "elev": "#F4F4F5",
        "hover": "#EFEFF1",
        "border": "#E4E4E7",
        "border_strong": "#D4D4D8",
        "text": "#0B0B0C",
        "muted": "#71717A",
        "faint": "#A1A1AA",
        "on_accent": "#FFFFFF",
        "shadow": "rgba(0,0,0,0.06)",
    },
    "dark": {
        "bg": "#0B0B0C",
        "surface": "#141416",
        "elev": "#1C1C1F",
        "hover": "#242427",
        "border": "#2A2A2E",
        "border_strong": "#3A3A3F",
        "text": "#F5F5F5",
        "muted": "#8A8A8F",
        "faint": "#5C5C62",
        "on_accent": "#0B0B0C",
        "shadow": "rgba(0,0,0,0.5)",
    },
}

# ATS semantics - the only chromatic colors in the interface.
SCORE_COLORS = {
    "excellent": "#059669",
    "good": "#0F766E",
    "fair": "#B45309",
    "low": "#DC2626",
    "matched": "#059669",
    "missing": "#DC2626",
    "warn": "#B45309",
}

AGENTS = [
    ("ResumeParserAgent", "Parses PDF into ontology"),
    ("RoleRecommenderAgent", "Formulates target roles"),
    ("JobSearchAgent", "Discovers live openings"),
    ("MatchGapAgent", "Scores ATS compatibility"),
    ("TailorAgent", "Rewrites for the JD"),
    ("VerifierAgent", "Audits for hallucinations"),
]

# Employment filters offered in the discovery stage. "Any" disables the
# filter; the rest are matched loosely against the feed's own spelling.
EMPLOYMENT_TYPES = [
    "Any",
    "Full Time",
    "Part Time",
    "Contract",
    "Internship",
]

# Label + gate for the primary "Analyse" control. The button always reflects
# the agent that is next in the pipeline, so its wording changes per agent:
#   (button label, agent index, session key that must be truthy to enable)
ANALYSE_ACTIONS = [
    ("Analyse Resume", 0, "resume_text"),
    ("Analyse Profile", 1, "profile"),
    ("Discover Jobs", 2, "selected_role"),
    ("Score ATS Match", 3, "discovered_jobs"),
    ("Tailor Resumes", 4, "discovered_jobs"),
    ("Re-verify Audits", 5, "tailored_resumes"),
]

# Why the control is unavailable, keyed by the same session state the action
# depends on. Shown as the button tooltip so the block is never a dead end.
ANALYSE_BLOCKERS = {
    0: "Upload a resume PDF first.",
    1: "Agent 1 must finish parsing your resume.",
    2: "Select a target role to search for openings.",
    3: "Run a job search to score matches.",
    4: "Discover jobs before tailoring a resume.",
    5: "Tailor at least one resume before re-auditing.",
}

NAV_H = 56
WORKFLOW_H = 50
BODY_TOP_PAD = NAV_H + WORKFLOW_H + 18


def theme_tokens() -> dict:
    return THEMES[st.session_state.theme_mode]


def esc(value) -> str:
    """Escape untrusted text before interpolating into raw HTML."""
    return _html.escape(str(value if value is not None else ""), quote=True)


# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "theme_mode": "dark",
        "resume_text": None,
        "profile": None,
        "role_recommendations": None,
        "selected_role": None,
        "search_keywords": [],
        "discovered_jobs": [],
        "match_reports": {},
        "tailored_resumes": {},
        "generated_pdfs": {},
        "uploaded_filename": None,
        "search_query": None,
        "search_location": "Remote",
        "employment_type": "Any",
        "look_back_days": 7,
        "search_history": [],
        "pending_history": None,
        "analyse_request": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def build_css() -> str:
    """Return the stylesheet with the active theme's tokens substituted in."""
    t = theme_tokens()
    score = SCORE_COLORS

    vars_css = "\n".join(
        "  --%s: %s;" % (k.replace("_", "-"), v) for k, v in t.items()
    )

    return (
        """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif !important; }
/* Safety net: anything Streamlit renders outside the containers themed below
   (portal menus, spinners, tooltips) otherwise inherits config.toml's dark
   text colour and disappears on the dark surface. */
body { color: var(--text); }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--faint); }

/* ── App frame ── */
.stApp { background: var(--bg) !important; }
.block-container {
    padding-top: %(body_pad)dpx !important;
    padding-bottom: 2rem !important;
    max-width: 100%% !important;
}
/* Streamlit's own chrome must be hidden and made transparent, otherwise the
   hard-coded header/toolbar overlays the app's own fixed nav bar. */
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"], [data-testid="stHeader"] {
    background: transparent !important; height: 0 !important;
    visibility: hidden !important;
}
[data-testid="stToolbar"], [data-testid="stToolbarActions"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMainBlockContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
}

/* The uploader and toolbar keep hard-coded dark surfaces in some builds.
   In 1.65 the size/type hint is a span (not a small element) and arrives with
   the light-theme text colour, which is unreadable on the dark dropzone, so
   the descendants are themed too rather than just the container.
   NB: no angle brackets in these comments. Streamlit hands the style block to
   an HTML parser, so a stray tag-looking token silently truncates the rest of
   the chunk it appears in. */
[data-testid="stFileUploaderDropzone"] div[role="button"],
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: var(--muted) !important; background: transparent !important;
}
[data-testid="stIconMaterial"] { color: var(--muted) !important; }

/* Streamlit's config.toml still pins a dark theme, so baseweb components
   that read theme colours directly (file uploader, segmented control) stay
   dark. Override those explicitly per mode. */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important; border-color: var(--border-strong) !important;
}
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzone"] > div > div {
    background: transparent !important; color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--muted) !important; }

/* ── Fixed navigation bar ── */
.app-nav {
    position: fixed; top: 0; left: 0; right: 0; height: %(nav_h)dpx;
    background: var(--bg); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.25rem; z-index: 1000;
}
/* st.html wraps content in a styled container; neutralise it so the fixed bar
   spans the full viewport instead of sitting in the page column. */
[data-testid="stHtml"] { margin: 0px !important; }
.nav-brand { display: flex; align-items: center; gap: .6rem; }
.nav-mark {
    width: 26px; height: 26px; border-radius: 7px; background: var(--text);
    color: var(--bg) !important; display: flex; align-items: center;
    justify-content: center; font-size: .7rem; font-weight: 700;
}
.nav-title { font-size: .93rem; font-weight: 600; color: var(--text) !important; letter-spacing: -.01em; }
.nav-sub { font-size: .68rem; color: var(--muted) !important; margin-left: .5rem; font-weight: 400; }
.nav-right { display: flex; align-items: center; gap: .9rem; }

.prov-group { display: flex; align-items: center; gap: .55rem; }
.prov {
    display: inline-flex; align-items: center; gap: .3rem;
    font-size: .68rem; color: var(--muted); font-weight: 500;
}
.prov-dot { width: 6px; height: 6px; border-radius: 50%%; background: var(--border-strong); }
.prov-dot.on { background: var(--text); }
.nav-mode {
    font-size: .68rem; font-weight: 600; color: var(--muted);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 3px 11px; background: var(--elev);
}

/* Segmented control (theme switch). Streamlit renders it as
   [data-testid="stButtonGroup"] > [role="radiogroup"] with aria-checked on
   the buttons, so target those rather than a stSegmentedControl id that no
   longer exists in 1.65. */
[data-testid="stButtonGroup"] { width: fit-content !important; }
[data-testid="stButtonGroup"] > [role="radiogroup"] {
    background: var(--elev) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; padding: 2px !important; gap: 2px !important;
    display: inline-flex !important;
}
[data-testid="stButtonGroup"] [role="radio"] {
    background: transparent !important; color: var(--muted) !important;
    font-size: .72rem !important; font-weight: 500 !important;
    border-radius: 6px !important; border: none !important; padding: 4px 14px !important;
}
[data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] {
    background: var(--text) !important; color: var(--on-accent) !important;
}
[data-testid="stWidgetLabel"] p { font-size: .7rem !important; color: var(--muted) !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: .07em; }

/* Do not force colour on button internals: the segmented control nests spans
   inside each button, and inheriting the button colour onto those spans makes
   the text disappear. Only the button itself is themed. */
[data-testid="stButtonGroup"] [role="radio"] > div,
[data-testid="stButtonGroup"] [role="radio"] > div span { color: inherit !important;
    background: transparent !important; }

/* ── Fixed agent workflow bar ── */
.agent-bar {
    position: fixed; top: %(nav_h)dpx; left: 0; right: 0; height: %(wf_h)dpx;
    background: var(--surface); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: .4rem; padding: 0 1.25rem;
    z-index: 999; overflow-x: auto; overflow-y: hidden;
}
.agent-pill {
    display: inline-flex; align-items: center; gap: .4rem; flex: 0 0 auto;
    padding: 5px 11px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--bg); font-size: .71rem; font-weight: 500;
    color: var(--faint); white-space: nowrap;
}
.agent-pill .ap-dot { width: 5px; height: 5px; border-radius: 50%%; background: var(--border-strong); flex: 0 0 auto; }
.agent-pill.done { color: var(--muted); border-color: var(--border-strong); }
.agent-pill.done .ap-dot { background: var(--muted); }
.agent-pill.active { color: var(--text); border-color: var(--text); background: var(--elev); font-weight: 600; }
.agent-pill.active .ap-dot { background: var(--text); width: 6px; height: 6px; }
.agent-arrow { color: var(--border-strong); font-size: .7rem; flex: 0 0 auto; user-select: none; }

/* ── Floating primary control ── */
/* The per-stage run button is the only control the user has to reach on every
   agent, and it lived at the top of the document beside the uploader. Its
   keyed container is lifted out of the column and pinned to the bottom-right
   so it stays reachable no matter where the page is scrolled. */
.st-key-analyse_fab {
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    z-index: 998;
    width: auto !important;
    min-width: 13rem;
}
/* Room for the floating button so it never covers the last row of content. */
.block-container { padding-bottom: 6rem !important; }

/* ── Panels / cards ── */
/* Side-panel frames are native bordered containers (see panel()). 1.65 puts
   the frame on the vertical block itself under an emotion hash that changes
   between builds, so it is selected by the panel heading it *directly*
   contains - only these cards have one, and the direct-child requirement keeps
   ancestor blocks from matching. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .panel-head) {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem !important;
}
.panel-head {
    font-size: .66rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: .08em; color: var(--muted); margin-bottom: .7rem;
    display: flex; align-items: center; justify-content: space-between;
}
.pill-row { margin-top: .7rem; }
.card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 10px; padding: .9rem 1rem; margin-bottom: .6rem;
}
.card:hover { border-color: var(--border-strong); background: var(--surface); }

/* ── Section heading ── */
.sec-head {
    display: flex; align-items: center; gap: .6rem; margin: 1.4rem 0 .8rem 0;
    padding: .55rem .7rem; border: 1px solid var(--border);
    border-radius: 10px; background: var(--surface);
}
.sec-num {
    width: 22px; height: 22px; border-radius: 6px; flex: 0 0 auto;
    border: 1px solid var(--border-strong); color: var(--muted);
    display: flex; align-items: center; justify-content: center;
    font-size: .66rem; font-weight: 600;
}
.sec-title { font-size: .93rem; font-weight: 600; color: var(--text); }
.sec-sub { font-size: .71rem; color: var(--muted); margin-top: 1px; }

/* ── Candidate profile ── */
.prof-name { font-size: 1.02rem; font-weight: 600; color: var(--text); letter-spacing: -.01em; }
.prof-badge {
    display: inline-block; margin-top: .3rem; padding: 2px 9px; border-radius: 999px;
    background: var(--elev); border: 1px solid var(--border);
    font-size: .68rem; color: var(--muted); font-weight: 500;
}
.prof-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-top: .8rem; }
.prof-cell { background: var(--elev); border: 1px solid var(--border); border-radius: 8px; padding: .45rem .6rem; }
.prof-cell .k { font-size: .6rem; color: var(--faint); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.prof-cell .v { font-size: .76rem; color: var(--text); font-weight: 500; margin-top: 1px; word-break: break-word; }
.prof-summary { font-size: .76rem; color: var(--muted); line-height: 1.55; margin-top: .8rem; }

/* ── Search history ── */
.hist-row {
    display: flex; align-items: center; justify-content: space-between; gap: .5rem;
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: .5rem .65rem; margin-bottom: .35rem;
}
.hist-q { font-size: .76rem; font-weight: 500; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-meta { font-size: .64rem; color: var(--faint); margin-top: 1px; }
.hist-score {
    font-size: .7rem; font-weight: 600; padding: 2px 7px; border-radius: 6px;
    background: var(--elev); border: 1px solid var(--border); flex: 0 0 auto;
}

/* ── Stage tracker ── */
.stage-track { display: flex; align-items: center; margin: .2rem 0 1rem 0; }
.stage-node { display: flex; align-items: center; gap: .35rem;
    font-size: .68rem; color: var(--faint); font-weight: 500; white-space: nowrap; }
.stage-node.done { color: var(--muted); }
.stage-node.active { color: var(--text); font-weight: 600; }
.stage-dot { width: 6px; height: 6px; border-radius: 50%%; background: var(--border-strong); }
.stage-node.done .stage-dot { background: var(--muted); }
.stage-node.active .stage-dot { background: var(--text); }
.stage-line { flex: 1; height: 1px; background: var(--border); margin: 0 .5rem; min-width: 12px; }
.stage-line.done { background: var(--border-strong); }

/* ── Job cards ── */
.job-title { font-size: .87rem; font-weight: 600; color: var(--text); }
.job-company { font-size: .74rem; color: var(--muted); margin-top: 1px; }
.job-meta { display: flex; gap: .8rem; flex-wrap: wrap; margin: .45rem 0 0 0; }
.job-meta-item { font-size: .69rem; color: var(--faint); }
.source-badge {
    display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 999px;
    background: var(--elev); border: 1px solid var(--border);
    font-size: .62rem; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .04em;
}

/* ── ATS score ring ── */
.ring-wrap { display: flex; flex-direction: column; align-items: center; gap: .4rem; }
.ring { width: 62px; height: 62px; border-radius: 50%%;
    display: flex; align-items: center; justify-content: center; }
.ring-inner {
    width: 47px; height: 47px; border-radius: 50%%; background: var(--bg);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring-val { font-size: .95rem; font-weight: 700; line-height: 1; }
.ring-lbl { font-size: .5rem; color: var(--faint); text-transform: uppercase;
    letter-spacing: .05em; margin-top: 1px; }

/* ── Pills ── */
.pill {
    display: inline-flex; align-items: center; background: var(--elev);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 2px 7px; font-size: .68rem; color: var(--text);
    margin: 2px 3px 2px 0; font-weight: 500;
}
.pill.matched { color: %(c_matched)s; border-color: %(c_matched)s; background: transparent; }
.pill.missing { color: %(c_missing)s; border-color: %(c_missing)s; background: transparent; }

/* ── Markdown ── */
/* Streamlit paints markdown with the text colour from config.toml, which is
   pinned to the light palette. On the dark surface that renders invisible but
   still occupies layout - the role analysis blockquote showed up as a ~120px
   empty gap - so markdown text is routed through the theme tokens. */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown strong,
.stMarkdown em, .stMarkdown a { color: var(--text) !important; }
.stMarkdown code {
    background: var(--elev) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 5px;
    padding: 1px 5px;
}
/* The segmented control renders its labels through the same markdown wrapper,
   so exclude it or the checked pill would put light text on a light fill. */
[data-testid="stButtonGroup"] .stMarkdown,
[data-testid="stButtonGroup"] .stMarkdown p { color: inherit !important; }

blockquote, .stMarkdown blockquote {
    border-left: 2px solid var(--border-strong) !important;
    padding: .15rem 0 .15rem .95rem !important;
    margin: .1rem 0 1.1rem 0 !important;
}
blockquote p, .stMarkdown blockquote p {
    color: var(--muted) !important; font-size: .84rem; line-height: 1.65;
}

/* ── Role cards ── */
/* Fixed line budgets (not min-heights) on the title, rationale and strength
   list, so every card in a row ends up exactly the same height and the Select
   buttons beneath them share a baseline. With min-height a card whose copy ran
   long simply grew, and the row came out ragged. */
.role-card {
    background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem 1.15rem; height: 100%%; display: flex; flex-direction: column;
    transition: border-color .15s ease;
}
.role-card:hover { border-color: var(--border-strong); }
.role-card.selected { border-color: var(--text); background: var(--elev); }
.role-score { font-size: 1.6rem; font-weight: 700; color: var(--text); line-height: 1; }
.role-title {
    font-size: .96rem; font-weight: 600; color: var(--text); line-height: 1.35;
    height: 2.7em; overflow: hidden;
}
.role-rationale {
    font-size: .8rem; color: var(--muted); margin-top: .45rem; line-height: 1.5;
    height: 7.5em; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical;
}
.role-strengths {
    margin-top: auto; padding-top: .7rem; border-top: 1px solid var(--border);
}
.role-strength { font-size: .79rem; color: var(--muted); padding: .16rem 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ── Buttons ── */
/* Descendant selectors, not a child combinator: passing help= to st.button
   makes Streamlit interpose a tooltip wrapper span between the container and
   the button, so the button stops matching and silently falls back to the
   unstyled light-theme default. */
.stButton button, .stDownloadButton button {
    border-radius: 8px !important; font-weight: 500 !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--bg) !important; color: var(--text) !important;
    font-size: .78rem !important;
}
/* Keep the tooltip wrapper out of the layout so width="stretch" still fills
   the column. */
[data-testid="stTooltipHoverTarget"] { display: block; width: 100%%; }
[data-testid="stTooltipHoverTarget"] > button { width: 100%% !important; }
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--text) !important; background: var(--elev) !important;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: var(--text) !important; color: var(--on-accent) !important;
    border-color: var(--text) !important;
}
/* Keep the primary fill on hover. Repainting it with --hover made the pill
   near-black against the dark background, so it appeared to vanish the moment
   the cursor reached it; dim slightly and add a ring instead. */
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {
    background: var(--text) !important; color: var(--on-accent) !important;
    filter: brightness(.92);
    box-shadow: 0 0 0 3px var(--elev);
}
/* The Analyse control sits in a narrow column beside the uploader, so keep
   its per-agent labels on one line. */
.stButton button { white-space: nowrap; }
/* A blocked control must still read as a control. Left on Streamlit's default
   it rendered transparent with heavily dimmed text, which vanished into the
   panel background; the blocked state now keeps a full-contrast border, a
   solid surface fill and a legible label. */
.stButton button:disabled, .stDownloadButton button:disabled {
    color: var(--muted) !important;
    background: var(--surface) !important;
    border-color: var(--border-strong) !important;
    opacity: 1 !important;
    cursor: not-allowed;
    box-shadow: inset 0 0 0 1px var(--border-strong);
}

/* ── Form controls ── */
/* config.toml pins Streamlit's light palette, so baseweb renders the native
   widgets with dark text on light surfaces and paints the value chips in the
   theme's red primary. On the dark theme that left the search row unreadable -
   the slider badge came out red on red, i.e. invisible - so every control is
   re-skinned onto the tokens here. The control surface is the input's own
   parent: baseweb's data-baseweb hooks are stripped from the DOM in 1.65. */
[data-testid="stTextInput"] div:has(> input),
[data-testid="stNumberInput"] div:has(> input),
[data-testid="stSelectbox"] div:has(> input),
[data-testid="stMultiSelect"] div:has(> input) {
    background: var(--bg) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input {
    color: var(--text) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--faint) !important; }
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg {
    color: var(--muted) !important;
}
/* Option list and the chips already chosen inside the multiselect. The list
   renders in a portal and its options only inherit the body colour, which is
   the light-theme dark grey, so the colour has to be set on the panel. */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stMultiSelectVirtualDropdown"] {
    background: var(--elev) !important; color: var(--text) !important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stMultiSelectVirtualDropdown"] [role="option"] {
    background: transparent !important; color: var(--text) !important;
}
[data-testid="stMultiSelectTagsContainer"] > span {
    background: var(--elev) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
}

/* ── Slider ── */
[data-testid="stSliderThumbValue"] {
    background: var(--text) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
[data-testid="stSliderThumbValue"] p { color: var(--on-accent) !important; }
[data-testid="stSliderTickBar"] p {
    color: var(--faint) !important; font-size: .68rem !important;
}
[data-testid="stSlider"] button { color: var(--text) !important; }
[data-testid="stSlider"] button svg circle { stroke: var(--text) !important; }
[data-testid="stSlider"] [role="group"] > div > div {
    background: var(--border-strong) !important; border-radius: 2px !important;
}

/* ── Inputs ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important; border: 1px dashed var(--border-strong) !important;
    border-radius: 10px !important; padding: 1.1rem !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--text) !important; }
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] summary { font-size: .78rem !important; color: var(--muted) !important; }
.stProgress > div > div > div > div { background: var(--text) !important; }

/* ── Audit / trace ── */
.audit-stat {
    display: flex; flex-direction: column; align-items: center;
    background: var(--elev); border: 1px solid var(--border);
    border-radius: 8px; padding: .6rem; text-align: center;
}
.audit-stat .val { font-size: .82rem; font-weight: 700; }
.audit-stat .lbl { font-size: .6rem; color: var(--faint); text-transform: uppercase;
    letter-spacing: .05em; margin-top: 2px; }
.trace-box {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 2px solid var(--text);
    color: var(--muted); font-family: 'SF Mono', Menlo, monospace;
    padding: .7rem .85rem; border-radius: 0 8px 8px 0;
    font-size: .68rem; line-height: 1.65;
}

/* ── Empty state ── */
.empty {
    border: 1px dashed var(--border-strong); border-radius: 10px;
    padding: 1.4rem 1rem; text-align: center; color: var(--muted);
    font-size: .76rem; line-height: 1.6; background: var(--surface);
}
.empty-mark { font-size: 1.3rem; opacity: .5; display: block; margin-bottom: .35rem; }

/* ── Misc ── */
.hr { height: 1px; background: var(--border); margin: 1.2rem 0; border: none; }
.delta-up {
    display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px;
    border-radius: 999px; font-size: .7rem; font-weight: 600;
    color: %(c_matched)s; border-color: %(c_matched)s;
}
.delta-flat {
    display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px;
    border-radius: 999px; font-size: .7rem; font-weight: 600;
    color: var(--muted); border: 1px solid var(--border-strong);
}
</style>
"""
        % {
            "body_pad": BODY_TOP_PAD,
            "nav_h": NAV_H,
            "wf_h": WORKFLOW_H,
            "c_matched": score["matched"],
            "c_missing": score["missing"],
        }
    ).replace(
        "</style>",
        ":root {\n%s\n}\n</style>" % vars_css,
        1,
    )


def inject_css() -> None:
    """
    Inject the stylesheet for the active theme.

    Three Streamlit constraints are worked around here:

    1. st.html() strips <style> blocks entirely in 1.65, so st.html() is
       used instead - it renders them verbatim.
    2. An oversized <style> block gets dropped, so the sheet is split into
       chunks under the observed limit and emitted separately.
    3. The chunk carrying the @import is the one that goes missing when it is
       packed with other rules, so the import is emitted on its own.
    """
    css = build_css()

    # Strip the wrapper tags; chunks are wrapped individually below.
    payload = css.strip()
    if payload.startswith("<style>"):
        payload = payload[len("<style>") :]
    if payload.endswith("</style>"):
        payload = payload[: -len("</style>")]

    # The :root token block is emitted on its own so a chunk boundary can never
    # split it and leave the variables undefined.
    root_match = re.search(r":root\s*\{.*?\}", payload, re.S)
    root_block = ""
    if root_match:
        root_block = root_match.group(0)
        payload = payload.replace(root_block, "", 1)
    if root_block:
        st.html("<style>%s</style>" % root_block)

    # Likewise for the font import. The URL contains semicolons (wght@300;400),
    # so this has to match the quoted url() token rather than run to the first
    # semicolon, which would leave a truncated string at the head of the next
    # chunk and get that whole <style> block discarded.
    import_match = re.search(r"@import\s+url\(['\"][^'\"]+['\"]\)\s*;", payload)
    if import_match:
        st.html("<style>%s</style>" % import_match.group(0))
        payload = payload.replace(import_match.group(0), "", 1)

    # Split only at top-level rule boundaries. Slicing on raw byte counts can
    # cut a selector list in half, which merges unrelated rules and makes the
    # wrong declarations apply to the wrong elements. A rule that is on its own
    # already over the cap is emitted regardless, since it cannot be split.
    chunk_size = 1800
    chunks, current, depth = [], [], 0
    for piece in re.split(r"(?<=\})", payload):
        if not piece:
            continue
        size = sum(len(c) for c in current) + len(piece)
        if current and size > chunk_size:
            chunks.append("".join(current))
            current = []
        current.append(piece)
        depth += piece.count("{") - piece.count("}")
        if depth <= 0:
            depth = 0
    if current:
        chunks.append("".join(current))

    for chunk in chunks:
        st.html("<style>%s</style>" % chunk)


inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score >= 80:
        return SCORE_COLORS["excellent"]
    if score >= 65:
        return SCORE_COLORS["good"]
    if score >= 50:
        return SCORE_COLORS["fair"]
    return SCORE_COLORS["low"]


def pipeline_stage() -> int:
    if st.session_state.tailored_resumes:
        return 5
    if st.session_state.discovered_jobs:
        return 4
    if st.session_state.selected_role:
        return 3
    if st.session_state.profile:
        return 2
    return 1


def active_agent_index() -> int:
    """0-based index of the agent currently doing work, for the fixed agent bar."""
    stage = pipeline_stage()
    if st.session_state.tailored_resumes:
        return 5
    if st.session_state.discovered_jobs:
        return 3
    if st.session_state.selected_role:
        return 2
    if st.session_state.profile:
        return 1
    return 0


def pending_agent_index() -> int:
    """
    Index of the agent whose work is the next actionable step.

    Distinct from active_agent_index(): that one only drives the workflow bar
    highlight, whereas this picks what the Analyse button should say and run.
    """
    if not st.session_state.profile:
        return 0
    if st.session_state.role_recommendations is None:
        return 1
    if not st.session_state.discovered_jobs:
        return 2
    if not st.session_state.match_reports:
        return 3
    if not st.session_state.tailored_resumes:
        return 4
    return 5


def analyse_action() -> tuple:
    """Return (label, agent index, tooltip) for the current Analyse button."""
    idx = pending_agent_index()
    label, agent, gate = ANALYSE_ACTIONS[idx]
    ready = gate is None or bool(st.session_state.get(gate))
    tip = "Run %s - %s" % (AGENTS[agent][0], AGENTS[agent][1])
    if not ready:
        tip = ANALYSE_BLOCKERS[idx]
    return label, agent, ready, tip


def run_parse_agent() -> None:
    """Agent 1: turn the uploaded PDF text into the typed profile."""
    text = st.session_state.resume_text
    if not text:
        return
    with st.spinner("Agent 1 parsing %s..." % st.session_state.uploaded_filename):
        st.session_state.profile = ResumeParserAgent().parse(text)


def run_role_agent() -> None:
    """Agent 2: recommend target roles from the parsed profile."""
    with st.spinner("Agent 2 formulating goal states..."):
        st.session_state.role_recommendations = RoleRecommenderAgent().recommend_roles(
            st.session_state.profile
        )


def run_match_agent() -> int:
    """
    Agent 4: score every discovered job for ATS fit.

    Returns the average score so callers can record it in search history.
    """
    jobs = st.session_state.discovered_jobs
    with st.spinner("Agent 4 running ATS analysis on %d jobs..." % len(jobs)):
        analyzer = MatchGapAgent()
        reports, failed = {}, []
        for job in jobs:
            try:
                reports[job.id] = analyzer.analyze_match(st.session_state.profile, job)
            except Exception as exc:
                failed.append(job.title or job.id)
                print("[Stage 4] %s: %s" % (job.title, exc))
        st.session_state.match_reports = reports

        scores = [r.overall_match_score for r in reports.values()]
        avg_ats = int(sum(scores) / len(scores)) if scores else 0

        if failed:
            st.error(
                "LLM analysis failed for %d of %d jobs (%s). Provider quotas may "
                "be exhausted - wait for reset or add credits."
                % (len(failed), len(jobs), ", ".join(failed[:3]))
            )
        elif reports:
            st.success("ATS analysis complete for %d jobs." % len(reports))

    return avg_ats


def run_job_agents(
    query: str, location: str, look_back_days: int, employment_type: str
) -> None:
    """Agent 3 discovers openings, then Agent 4 scores them."""
    with st.spinner("Agent 3 searching %s..." % query):
        jobs = JobSearchAgent().discover_jobs(
            query,
            location=location,
            target_count=5,
            max_age_hours=max(1, int(look_back_days)) * 24,
            employment_type=employment_type,
            extra_queries=st.session_state.get("search_keywords") or None,
        )
    st.session_state.discovered_jobs = jobs

    if not jobs:
        st.error("No openings returned for %s. Try a broader query." % query)
        return

    avg_ats = run_match_agent()

    import datetime as _dt

    st.session_state.search_history.append(
        {
            "query": query,
            "location": location,
            "job_count": len(jobs),
            "avg_ats": avg_ats,
            "timestamp": _dt.datetime.now().strftime("%H:%M:%S"),
        }
    )


def run_tailor_agents(job_ids: list) -> None:
    """Agents 5 and 6: rewrite each resume, audit it, compile the PDF."""
    orchestrator = TailoringOrchestrator()
    trace_ph = st.empty()
    traces = []

    def trace_callback(msg: str) -> None:
        traces.append(msg)
        trace_ph.html(
            '<div class="trace-box"><b>Agent trace</b><br/>%s</div>'
            % "<br/>".join(esc(t) for t in traces[-5:])
        )

    progress = st.progress(0)
    for idx, jid in enumerate(job_ids):
        job = next((j for j in st.session_state.discovered_jobs if j.id == jid), None)
        if not job:
            continue
        report = st.session_state.match_reports.get(jid)
        if not report:
            report = MatchGapAgent().analyze_match(st.session_state.profile, job)
            st.session_state.match_reports[jid] = report

        tailored, audit, cycles = orchestrator.run_tailoring_pipeline(
            profile=st.session_state.profile,
            job=job,
            match_report=report,
            status_callback=trace_callback,
        )
        pdf_bytes = compile_tailored_pdf(tailored)
        st.session_state.tailored_resumes[jid] = {
            "resume": tailored,
            "audit": audit,
            "cycles": cycles,
        }
        st.session_state.generated_pdfs[jid] = pdf_bytes
        progress.progress((idx + 1) / len(job_ids))
    st.success("All resumes synthesized, verified and compiled.")


def render_agent_bar() -> None:
    """Fixed workflow bar listing all six agents with live status."""
    stage = pipeline_stage()
    current = active_agent_index()

    pills = []
    for i, (name, _desc) in enumerate(AGENTS):
        if i < current:
            cls = "done"
        elif i == current:
            cls = "active"
        else:
            cls = ""
        pills.append(
            '<div class="agent-pill %s"><span class="ap-dot"></span>%s</div>'
            % (cls, esc(name))
        )
        if i < len(AGENTS) - 1:
            pills.append('<span class="agent-arrow">&rarr;</span>')

    st.html('<div class="agent-bar">%s</div>' % "".join(pills))


def render_nav_bar() -> None:
    """Fixed top navigation: brand, provider status, theme switch."""
    mode = st.session_state.theme_mode
    prov = [
        ("OpenRouter", settings.is_openrouter_configured()),
        ("Groq", settings.is_groq_configured()),
        ("Gemini", settings.is_gemini_configured()),
    ]
    prov_html = "".join(
        '<span class="prov"><span class="prov-dot %s"></span>%s</span>'
        % ("on" if ok else "", esc(label))
        for label, ok in prov
    )

    st.html("""
<div class="app-nav">
  <div class="nav-brand">
    <div class="nav-mark">◆</div>
    <div>
      <span class="nav-title">AI Job Search &amp; Resume Tailoring</span>
      <span class="nav-sub">Multi-Agent System</span>
    </div>
  </div>
  <div class="nav-right">
    <div class="prov-group">%(prov)s</div>
    <span class="nav-mode">%(mode)s</span>
  </div>
</div>
"""
        % {"prov": prov_html, "mode": esc(mode.capitalize())},)


def render_theme_switch() -> None:
    """
    Real Light/Dark control.

    This has to be a native Streamlit widget: a raw <button> emitted inside
    unsafe_allow_html cannot trigger a re-run, so it would be inert. It is
    therefore rendered in normal flow (top of the left pane) rather than
    floated into the fixed nav bar with CSS.
    """
    mode = st.session_state.theme_mode
    picked = st.segmented_control(
        "Appearance",
        options=list(THEMES.keys()),
        default=mode,
        key="theme_picker",
    )
    if picked in THEMES and picked != mode:
        st.session_state.theme_mode = picked
        st.rerun()


@contextmanager
def panel(title: str, meta: str = ""):
    """
    Context manager for a side-panel card.

    A hand-written '<div class="panel">' split across several st.html() calls
    does not nest: each call emits its own fragment, so the opening tag renders
    as an empty bordered box and the stray '</div>' as a second one. Native
    bordered containers keep the frame and its widgets inside one real box.
    """
    with st.container(border=True):
        st.html('<div class="panel-head"><span>%s</span><span>%s</span></div>'
            % (esc(title), esc(meta)),)
        yield


def clamp(text: str, limit: int) -> str:
    """Trim to a word boundary, so clipped copy never ends mid-word."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.-") + "…"


def render_side_panel() -> None:
    """Left 3/7 pane: appearance switch, candidate details, search history."""
    render_theme_switch()

    profile = st.session_state.profile
    with panel("Candidate"):
        if not profile:
            st.html(
                '<div class="empty"><span class="empty-mark">◇</span>'
                "Upload a resume to build the candidate profile."
                "</div>",)
        else:
            contact = profile.contact
            skills = profile.skills.all_technical_skills()
            st.html("""
<div class="prof-name">%(name)s</div>
<div class="prof-badge">%(level)s</div>
<div class="prof-grid">
  <div class="prof-cell"><div class="k">Experience</div><div class="v">%(exp)s yrs</div></div>
  <div class="prof-cell"><div class="k">Skills</div><div class="v">%(nskills)s</div></div>
  <div class="prof-cell"><div class="k">Email</div><div class="v">%(email)s</div></div>
  <div class="prof-cell"><div class="k">Phone</div><div class="v">%(phone)s</div></div>
  <div class="prof-cell"><div class="k">Projects</div><div class="v">%(nproj)s</div></div>
  <div class="prof-cell"><div class="k">Education</div><div class="v">%(nedu)s</div></div>
</div>
<div class="prof-summary">%(summary)s</div>
"""
                % {
                    "name": esc(contact.name),
                    "level": esc(profile.inferred_seniority_level),
                    "exp": esc(profile.total_years_experience),
                    "nskills": len(skills),
                    "email": esc(contact.email or "-"),
                    "phone": esc(contact.phone or "-"),
                    "nproj": len(profile.projects),
                    "nedu": len(profile.education),
                    "summary": esc(profile.professional_summary),
                },)

            pills = "".join('<span class="pill">%s</span>' % esc(s) for s in skills)
            st.html('<div class="pill-row">%s</div>'
                % (pills or '<span class="pill">-</span>'),)

    # ── Search history ──
    history = st.session_state.search_history
    with panel("Search History", str(len(history))):
        if not history:
            st.html('<div class="empty" style="padding:1rem .8rem">'
                "No searches yet.<br/>Results appear here as you search."
                "</div>",)
        else:
            for idx in range(len(history) - 1, -1, -1):
                h = history[idx]
                avg = h.get("avg_ats") or 0
                st.html("""
<div class="hist-row" id="hist-%(i)d">
  <div style="min-width:0;flex:1">
    <div class="hist-q">%(q)s</div>
    <div class="hist-meta">%(n)s jobs &middot; %(when)s</div>
  </div>
  <div class="hist-score" style="color:%(color)s">%(avg)s</div>
</div>
"""
                % {
                    "i": idx,
                    "q": esc(h["query"]),
                    "n": h["job_count"],
                    "when": esc(h["timestamp"]),
                    "avg": avg if avg else "-",
                    "color": score_color(avg) if avg else "var(--muted)",
                },)
                if st.button(
                    "Restore",
                    key=f"restore_{idx}_{h['timestamp']}",
                    width="stretch",
                ):
                    st.session_state.pending_history = h
                    st.rerun()

    # ── Controls ──
    tool = JobSearchTool()
    stats = tool.cache_stats()
    with panel("Cache", "%d jobs" % stats["total_cached_jobs"]):
        if st.button("Refresh Job Cache", width="stretch"):
            with st.spinner("Fetching latest jobs..."):
                tool.refresh_cache(
                    st.session_state.search_query
                    or st.session_state.selected_role
                    or "Software Engineer",
                    max_results=25,
                )
            st.rerun()

        if st.button("Reset Pipeline", width="stretch"):
            keep_theme = st.session_state.theme_mode
            st.session_state.clear()
            st.session_state.theme_mode = keep_theme
            st.rerun()


def render_stage_track() -> None:
    """Compact stage tracker shown at the top of the right pane."""
    stage = pipeline_stage()
    stages = [
        "Parse",
        "Roles",
        "Search",
        "Score",
        "Tailor",
    ]
    parts = []
    for i, label in enumerate(stages, start=1):
        if i < stage:
            cls = "done"
        elif i == stage:
            cls = "active"
        else:
            cls = ""
        parts.append(
            '<div class="stage-node %s"><span class="stage-dot"></span>%s</div>'
            % (cls, esc(label))
        )
        if i < len(stages):
            parts.append('<div class="stage-line %s"></div>' % ("done" if i < stage else ""))
    st.html('<div class="stage-track">%s</div>' % "".join(parts),)


def section(num: int, title: str, sub: str = "") -> None:
    # Rendered with a single HTML block: st.html() wraps its output in
    # <p> tags, which visually breaks the badge / title / subtitle row apart.
    st.html(
        """
<div class="sec-head">
  <div class="sec-num">%(n)d</div>
  <div>
    <div class="sec-title">%(title)s</div>
    <div class="sec-sub">%(sub)s</div>
  </div>
</div>
"""
        % {"n": num, "title": esc(title), "sub": esc(sub)}
    )


def render_empty(icon: str, text: str) -> None:
    st.html(
        '<div class="empty"><span class="empty-mark">%s</span>%s</div>'
        % (icon, esc(text))
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIXED CHROME
# ─────────────────────────────────────────────────────────────────────────────
render_nav_bar()
render_agent_bar()

pending = st.session_state.pop("pending_history", None)
if pending:
    st.session_state.search_query = pending["query"]
    st.session_state.search_location = pending.get("location", "Remote")


# ─────────────────────────────────────────────────────────────────────────────
# 3:7 SPLIT
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 7], gap="medium")

with left_col:
    render_side_panel()

with right_col:
    render_stage_track()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 - RESUME INGESTION (Agent 1)
    # ─────────────────────────────────────────────────────────────────────────
    section(1, "Resume Ingestion", "Agent 1 parses your PDF into a typed ontology")

    c_upload, c_analyse = st.columns([5, 2], gap="small", vertical_alignment="center")
    with c_upload:
        uploaded = st.file_uploader("Resume (PDF)", type=["pdf"], label_visibility="collapsed")
        # Ingest inside this column so the text is in session state *before*
        # the Analyse button renders. Doing it afterwards left the control
        # drawn from the previous run's state, i.e. disabled for one cycle.
        if uploaded is not None and st.session_state.uploaded_filename != uploaded.name:
            with st.spinner("Extracting text from %s..." % uploaded.name):
                text = extract_text_from_pdf(uploaded)
            st.session_state.resume_text = text
            st.session_state.uploaded_filename = uploaded.name
            st.session_state.profile = None
            st.session_state.role_recommendations = None
            st.session_state.selected_role = None
            st.session_state.discovered_jobs = []
            st.session_state.match_reports = {}
            st.session_state.tailored_resumes = {}
            st.session_state.generated_pdfs = {}
            if not text.strip():
                st.error(
                    "No selectable text found in %s. If it is a scanned "
                    "image, export a text-based PDF and re-upload." % uploaded.name
                )
    with c_analyse:
        label, agent, ready, tip = analyse_action()
        # The control is keyed so the stylesheet can pull it out of the column
        # and pin it to the bottom-right of the viewport; the click handling
        # itself is unchanged.
        with st.container(key="analyse_fab"):
            if st.button(
                label,
                key="analyse_run",
                type="primary",
                disabled=not ready,
                width="stretch",
                help=tip,
            ):
                st.session_state.analyse_request = agent

    # The click above only records intent. Running it here (rather than inline)
    # keeps each stage's inputs in scope for the action it triggers, and avoids
    # calling st.rerun() from inside a button handler.
    request = st.session_state.pop("analyse_request", None)

    if request == 0:
        run_parse_agent()
        st.rerun()

    if not st.session_state.profile:
        render_empty("◇", "No resume loaded yet. Upload a PDF and run Analyse Resume.")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 - ROLE RECOMMENDATIONS (Agent 2)
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.profile:
        st.html('<hr class="hr">')
        section(2, "Career Strategy", "Agent 2 formulates your target roles")

        if request == 1 and st.session_state.role_recommendations is None:
            run_role_agent()
            st.rerun()

        if st.session_state.role_recommendations:
            resp = st.session_state.role_recommendations
            st.markdown("> %s" % esc(resp.candidate_summary_analysis))

            # Three per row: any narrower and the rationale wraps into a tall
            # column that leaves the Select buttons on different baselines.
            roles = resp.recommended_roles
            for start in range(0, len(roles), 3):
                row = roles[start : start + 3]
                cols = st.columns(len(row))
                for col, role in zip(cols, row):
                    n = start + row.index(role)
                    with col:
                        selected = st.session_state.selected_role == role.role_title
                        strengths = "".join(
                            '<div class="role-strength">%s%s</div>'
                            % ("&middot;&nbsp;", esc(clamp(s, 34)))
                            for s in role.key_matching_strengths[:4]
                        )
                        st.html(
                            """
<div class="role-card %(sel)s">
  <div class="role-score">%(score)d%%</div>
  <div class="role-title">%(title)s</div>
  <div class="role-rationale">%(rationale)s</div>
  <div class="role-strengths">%(strengths)s</div>
</div>
"""
                            % {
                                "sel": "selected" if selected else "",
                                "score": role.fit_score,
                                "title": esc(role.role_title),
                                "rationale": esc(clamp(role.rationale, 220)),
                                "strengths": strengths,
                            },)
                        label = "Selected" if selected else "Select Role %d" % (n + 1)
                        if st.button(
                            label,
                            key="sel_role_%d" % n,
                            width="stretch",
                            type="primary" if selected else "secondary",
                        ):
                            st.session_state.selected_role = role.role_title
                            # Seed the query with the keywords Agent 2 already
                            # produced for this role. They were previously
                            # generated and then discarded, leaving the search
                            # running on the bare role title.
                            st.session_state.search_query = (
                                role.recommended_search_keywords[0]
                                if role.recommended_search_keywords
                                else role.role_title
                            )
                            st.session_state.search_keywords = list(
                                role.recommended_search_keywords
                            )
                            st.session_state.discovered_jobs = []
                            st.session_state.match_reports = {}
                            st.session_state.tailored_resumes = {}
                            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 - JOB DISCOVERY + GAP ANALYSIS (Agents 3 & 4)
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.selected_role:
        st.html('<hr class="hr">')
        section(
            3,
            "Job Discovery - %s" % st.session_state.selected_role,
            "Agent 3 searches live feeds, Agent 4 scores ATS compatibility",
        )

        # Four controls on one baseline. The two columns that hold a native
        # widget with its own label box (selectbox, slider) are given the same
        # width as the text inputs so the labels sit on a single line.
        c_query, c_loc, c_type, c_days = st.columns([3, 2, 2, 2])
        with c_query:
            # `or` rather than get(): a session value of None must fall through
            # to the selected role instead of rendering an empty box.
            default_query = (
                st.session_state.get("search_query") or st.session_state.selected_role
            )
            query = st.text_input("Search query", value=default_query, key="q_input")
            st.session_state.search_query = query
        with c_loc:
            location = st.text_input(
                "Location", value=st.session_state.get("search_location", "Remote"),
                key="loc_input", placeholder="City, state or Remote",
            )
            st.session_state.search_location = location
        with c_type:
            employment_type = st.selectbox(
                "Employment type",
                options=EMPLOYMENT_TYPES,
                key="type_input",
            )
            st.session_state.employment_type = employment_type
        with c_days:
            # Defaults to a week rather than day one: at the very first stop the
            # thumb sits on the left edge and Streamlit clips the value badge
            # in half, which reads as a rendering fault.
            look_back = st.slider("Look-back", 1, 30, 7, key="days_input",
                                  help="How many days back to search for postings.")
            st.session_state.look_back_days = look_back

        # Agents 3 and 4 run together from the single Analyse control above.
        # Agent 4 also runs alone when a previous search came back unscored.
        if request == 2:
            run_job_agents(query, location, look_back, employment_type)
            st.rerun()
        elif request == 3:
            run_match_agent()
            st.rerun()

        jobs = st.session_state.discovered_jobs
        if not jobs:
            render_empty("◇", "No jobs discovered yet. Use Discover Jobs to populate this stage.")
        else:
            st.html("**%d** active positions found." % len(jobs)
            )
            for job in jobs:
                report = st.session_state.match_reports.get(job.id)
                score = report.overall_match_score if report else 0
                color = score_color(score)
                deg = int(score * 3.6)

                c_info, c_ring = st.columns([4, 1])
                with c_info:
                    skills = "".join(
                        '<span class="pill">%s</span>' % esc(s)
                        for s in (job.required_skills or [])[:8]
                    )
                    st.html(
                        """
<div class="card">
  <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap">
    <span class="job-title">%(title)s</span>
    <span class="source-badge">%(portal)s</span>
  </div>
  <div class="job-company">at %(company)s</div>
  <div class="job-meta">
    <span class="job-meta-item">%(loc)s</span>
    <span class="job-meta-item">%(etype)s</span>
  </div>
  <div style="margin-top:.45rem">%(skills)s</div>
</div>
"""
                        % {
                            "title": esc(job.title),
                            "portal": esc(job.portal_name),
                            "company": esc(job.company),
                            "loc": esc(job.location),
                            "etype": esc(job.employment_type),
                            "skills": skills,
                        },)
                    if job.apply_link:
                        st.html('<a href="%s" target="_blank" style="font-size:.7rem;'
                            'color:var(--muted)">Apply &nearr;</a>'
                            % esc(job.apply_link),)

                    if report:
                        matched = "".join(
                            '<span class="pill matched">%s</span>' % esc(s)
                            for s in report.matched_skills
                        )
                        missing = "".join(
                            '<span class="pill missing">%s</span>' % esc(s)
                            for s in report.missing_skills
                        )
                        c_m, c_x = st.columns(2)
                        with c_m:
                            st.html("**Matched**")
                            st.html(matched or '<span class="pill">-</span>')
                        with c_x:
                            st.markdown("**Gaps**")
                            st.html(missing or '<span class="pill">-</span>')
                        with st.expander("Recommendations & description"):
                            for rec in report.tailoring_recommendations:
                                st.markdown("- %s" % esc(rec))
                            st.markdown("---")
                            st.text((job.description or "")[:600])

                with c_ring:
                    st.html(
                        """
<div class="ring-wrap">
  <div class="ring" style="background:conic-gradient(%(color)s %(deg)ddeg, var(--elev) 0deg)">
    <div class="ring-inner">
      <span class="ring-val" style="color:%(color)s">%(score)d</span>
      <span class="ring-lbl">ATS</span>
    </div>
  </div>
</div>
"""
                        % {"color": color, "deg": deg, "score": score},)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4 - TAILORING & AUDIT (Agents 5 & 6)
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.discovered_jobs:
        st.html('<hr class="hr">')
        section(
            4,
            "Autonomous Tailoring & Audit",
            "Agent 5 rewrites your resume, Agent 6 verifies zero hallucinations",
        )

        selected_job_ids = st.multiselect(
            "Select jobs to tailor for",
            options=[j.id for j in st.session_state.discovered_jobs],
            default=[st.session_state.discovered_jobs[0].id],
            format_func=lambda jid: next(
                "%s @ %s" % (j.title, j.company)
                for j in st.session_state.discovered_jobs
                if j.id == jid
            ),
        )

        # Agents 5 and 6. Triggered either by this button or by the Analyse
        # control at the top of the column, which routes the same selection.
        # A re-audit (request 5) falls back to whatever was tailored already.
        tailor_targets = selected_job_ids or list(st.session_state.tailored_resumes)
        if st.button(
            "Tailor Selected & Audit",
            type="primary",
            disabled=not selected_job_ids,
            width="stretch",
        ) or (request in (4, 5) and tailor_targets):
            run_tailor_agents(tailor_targets)
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5 - PDF DELIVERY
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.tailored_resumes:
        st.html('<hr class="hr">')
        section(5, "Tailored Resume Downloads", "Verified, grounded, ATS-optimized PDFs")

        for jid, data in st.session_state.tailored_resumes.items():
            job = next((j for j in st.session_state.discovered_jobs if j.id == jid), None)
            if not job:
                continue
            resume = data["resume"]
            audit = data["audit"]
            cycles = data["cycles"]
            pdf_bytes = st.session_state.generated_pdfs.get(jid)
            delta = resume.ats_score_after - resume.ats_score_before
            approved = audit.status == "APPROVED"
            delta_cls = "delta-up" if delta > 0 else "delta-flat"

            st.html("""
<div class="card">
  <div class="job-title">%(name)s &rarr; %(job)s</div>
  <div class="job-company">at %(company)s</div>
  <div style="display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin-top:.55rem">
    <span class="job-meta-item">ATS %(before)d%% &rarr; %(after)d%%</span>
    <span class="%(delta_cls)s">%(sign)s%(delta)d%%</span>
    <span class="job-meta-item" style="color:%(audit_color)s">%(icon)s %(audit)s</span>
    <span class="job-meta-item">%(cycles)d cycle(s)</span>
  </div>
</div>
"""
                % {
                    "name": esc(resume.candidate_name),
                    "job": esc(job.title),
                    "company": esc(job.company),
                    "before": resume.ats_score_before,
                    "after": resume.ats_score_after,
                    "delta_cls": delta_cls,
                    "sign": "+" if delta > 0 else "",
                    "delta": delta,
                    "audit_color": SCORE_COLORS["matched"] if approved else SCORE_COLORS["missing"],
                    "icon": "&#10003;" if approved else "&#10007;",
                    "audit": esc(audit.status),
                    "cycles": cycles,
                },)

            if pdf_bytes:
                st.download_button(
                    "Download ATS PDF",
                    data=pdf_bytes,
                    file_name="Resume_%s_%s.pdf"
                    % (resume.candidate_name.replace(" ", "_"), job.company.replace(" ", "_")),
                    mime="application/pdf",
                    key="down_pdf_%s" % jid,
                    width="stretch",
                    type="primary",
                )

            with st.expander("Inspect tailored content & verifier audit"):
                t1, t2 = st.columns(2)
                with t1:
                    st.html("**Summary**")
                    st.write(resume.professional_summary)
                    st.markdown("**Prioritized skills**")
                    pills = "".join(
                        '<span class="pill">%s</span>' % esc(s)
                        for s in resume.prioritized_skills.all_technical_skills()
                    )
                    st.html(pills)
                with t2:
                    st.markdown("**Verifier report**")
                    st.html(
                        """
<div style="display:flex;gap:.5rem;margin-bottom:.6rem">
  <div class="audit-stat" style="flex:1">
    <div class="val" style="color:%(g_color)s">%(grounded)s</div>
    <div class="lbl">Grounding</div>
  </div>
  <div class="audit-stat" style="flex:1">
    <div class="val">%(risk)s</div>
    <div class="lbl">Risk</div>
  </div>
</div>
"""
                        % {
                            "g_color": SCORE_COLORS["matched"]
                            if audit.is_grounded
                            else SCORE_COLORS["missing"],
                            "grounded": "Grounded" if audit.is_grounded else "Not grounded",
                            "risk": esc(audit.hallucination_risk_level),
                        },)
                    st.markdown("*%s*" % esc(audit.audit_notes))
                    if audit.issues_found:
                        st.warning(
                            "%d issue(s) detected during audit." % len(audit.issues_found)
                        )

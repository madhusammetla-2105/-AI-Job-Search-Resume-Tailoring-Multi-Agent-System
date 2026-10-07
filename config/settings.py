"""
Central configuration module for the Multi-Agent System.
Loads environment variables and exposes typed application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application configuration and environment manager."""

    # Project Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = BASE_DIR / "output_resumes"

    # SQLite cache for Agent 3 (JobSearchTool)
    SQLITE_DB_PATH: str = str(BASE_DIR / "data" / "job_cache.db")

    # ── OpenRouter (Primary LLM – free models, OpenAI-compatible) ────────────
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    # Free models tried in priority order. Every ID below is a verified $0
    # endpoint in the OpenRouter catalog (checked against /api/v1/models).
    #
    # Position 1 is deliberately the 550B MoE for maximum reasoning quality.
    # Measured latency on a role-recommendation call (509 prompt tokens):
    #   nemotron-3-ultra-550b-a55b ~64s   nemotron-3-super-120b-a12b ~10.5s
    #   liquid/lfm-2.5-2.6b              ~7.1s  nemotron-3.5-lightning ~124s
    # The fallback order below is speed-ranked, so when the primary is
    # rate-limited the chain degrades quickly instead of hitting another slow
    # endpoint. Swap positions 1 and 2 for a ~6x faster cold run.
    OPENROUTER_FREE_MODELS: list = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",   # primary: 550B-A55B MoE, 262K ctx
        "liquid/lfm-2.5-2.6b:free",                 # fast fallback (~7s)
        "nvidia/nemotron-3-super-120b-a12b:free",   # 120B-A12B MoE, 262K ctx (~10s)
        "google/gemma-4-31b-it:free",               # text + image + video, 262K ctx
        "google/gemma-4-26b-a4b-it:free",           # retained working fallback
        "nvidia/nemotron-3.5-lightning:free",       # slowest measured (~124s): last
    ]
    # Excluded on purpose (verified against the OpenRouter catalog):
    #   - Paid-only, no :free endpoint -> bare IDs would bill OpenRouter credits:
    #       openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3-coder,
    #       qwen/qwen3-next-80b-a3b-instruct, nvidia/nemotron-3-nano-30b-a3b,
    #       meta-llama/llama-3.3-70b-instruct, meta-llama/llama-3.2-3b-instruct,
    #       nousresearch/hermes-3-llama-3.1-405b,
    #       cognitivecomputations/dolphin-mistral-24b-venice-edition
    #   - tencent/hy3: paid-only and flagged for deprecation (July 2026).
    #   - Not present in the catalog at all, so every attempt would fail:
    #       nvidia/nemotron-nano-9b-v2, liquid/lfm-2.5-1.2b-instruct,
    #       liquid/lfm-2.5-1.2b-thinking

    # ── Groq (Secondary LLM – fast inference, OpenAI-compatible) ────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODELS: list = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    # ── Gemini (Tertiary LLM – Google AI Studio) ───────────────────────────────
    GEMINI_MODELS: list = [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ]

    # LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gemini-3.5-flash").strip()
    MAX_TOKENS: int = 8192  # ceiling used when an agent supplies no explicit cap
    TEMPERATURE: float = 0.2  # Low temperature for factual, deterministic resume tailoring

    # Per-agent output ceilings. Generation time scales with the number of
    # tokens actually emitted, and slower models will happily run to a high cap
    # instead of stopping, so each agent is capped near its observed usage.
    # A cap that is too low truncates JSON and triggers a costly repair call,
    # so these leave headroom rather than hugging the measured output length.
    AGENT_MAX_TOKENS: dict = {
        "ResumeParserAgent": 3000,
        "RoleRecommenderAgent": 1500,
        "JobSearchAgent": 1024,      # makes no LLM calls today; safety cap only
        "MatchGapAgent": 2000,
        "TailorAgent": 2500,
        "VerifierAgent": 2000,
    }

    # Retry / backoff controls for transient provider errors (429 / 5xx)
    MAX_RETRIES_PER_MODEL: int = 3
    RETRY_BACKOFF_SECONDS: float = 4.0
    # Free-tier endpoints are rate-limited per minute; retrying one immediately
    # just burns wall-clock (measured ~12s per skipped model for 3 attempts plus
    # 4s/8s backoff). When enabled, a free-endpoint 429 advances to the next
    # model after a single attempt instead.
    FAST_FAIL_FREE_RATE_LIMITS: bool = True

    # Agent Loop Controls
    MAX_VERIFICATION_RETRIES: int = 2
    MIN_ACCEPTABLE_ATS_SCORE: int = 65

    def is_gemini_configured(self) -> bool:
        """Check if a Gemini API key is present (non-empty)."""
        return bool(self.GEMINI_API_KEY and len(self.GEMINI_API_KEY.strip()) > 10)

    def is_openrouter_configured(self) -> bool:
        """Check if an OpenRouter API key is present."""
        return bool(self.OPENROUTER_API_KEY and self.OPENROUTER_API_KEY.startswith("sk-or-"))

    def is_groq_configured(self) -> bool:
        """Check if a Groq API key is present (starts with gsk_)."""
        return bool(self.GROQ_API_KEY and self.GROQ_API_KEY.startswith("gsk_"))

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Ensure data dir exists for the SQLite job cache
        Path(self.SQLITE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)


# Global singleton settings instance
settings = Settings()
settings.ensure_directories()

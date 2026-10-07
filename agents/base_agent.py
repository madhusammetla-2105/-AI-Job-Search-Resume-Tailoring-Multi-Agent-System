"""
BaseAgent – Multi-provider LLM orchestration layer.

Provider priority (highest → lowest):
  1. OpenRouter free models  (no credits, OpenAI-compatible REST)
  2. Groq                    (fast inference, free tier, OpenAI-compatible)
  3. Google Gemini           (if key is valid)

Academic Alignment: Module IX – Multi-Agent Coordination & Reflection.
"""

import json
import re
import time
import warnings
import requests
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

# Suppress SDK deprecation notices for clean terminal UI logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
except ImportError:
    google_genai = None
    genai_types = None

from config.settings import settings

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Sentinel exceptions
# ---------------------------------------------------------------------------

class _CreditExhaustedError(Exception):
    """Raised when a provider rejects a request due to insufficient credits."""

class _RateLimitError(Exception):
    """Raised when a provider's rate limit is hit (429)."""


# Markers that indicate a NON-transient, session-long quota exhaustion.
# Retrying these wastes wall-clock time and never succeeds, so the provider is
# tripped out of the failover chain for the rest of the process lifetime.
_PERMANENT_QUOTA_MARKERS = (
    "free-models-per-day",
    "free-models-per-minute",
    "requests per day",
    "monthly limit",
    "daily limit",
    "add 10 credits",
)


def _is_permanent_quota_exhaustion(body: str) -> bool:
    body = (body or "").lower()
    return any(marker in body for marker in _PERMANENT_QUOTA_MARKERS)


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Abstract base class providing structured LLM prompt execution with
    automatic multi-provider failover.

    Failover chain:
        OpenRouter (free)  →  Groq (free tier)  →  Gemini

    A provider whose quota is permanently exhausted (e.g. a daily free-tier cap)
    is tripped out of the chain for the lifetime of the process.
    """

    # Class-level circuit breaker shared by every agent instance in the process.
    _tripped_providers: set = set()

    def __init__(self, agent_name: str, model_name: Optional[str] = None):
        self.agent_name = agent_name
        self.model_name = model_name or settings.DEFAULT_LLM_MODEL
        self._init_llm_clients()

    def _trip_provider(self, provider: str) -> None:
        """Disable a provider for the remainder of the process."""
        BaseAgent._tripped_providers.add(provider)
        attr = f"{provider.lower()}_available"
        if hasattr(self, attr):
            setattr(self, attr, False)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_llm_clients(self) -> None:
        """Detect and initialise available LLM providers."""
        self.openrouter_available = settings.is_openrouter_configured()
        self.groq_available       = settings.is_groq_configured()
        self.gemini_available     = settings.is_gemini_configured()

        # Honour the process-wide circuit breaker from earlier calls.
        for provider in BaseAgent._tripped_providers:
            attr = f"{provider.lower()}_available"
            if hasattr(self, attr):
                setattr(self, attr, False)

        if self.gemini_available:
            try:
                if google_genai is None:
                    raise ImportError("google-genai package not installed.")
                self.gemini_client = google_genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as exc:
                print(f"[{self.agent_name}] ⚠️  Gemini init failed: {exc}")
                self.gemini_available = False

        # Log active providers once at startup
        providers = []
        if self.openrouter_available: providers.append("OpenRouter ✅")
        if self.groq_available:       providers.append("Groq ✅")
        if self.gemini_available:     providers.append("Gemini ✅")
        if not providers:             providers.append("No active providers ⚠️")
        print(f"[{self.agent_name}] Active LLM providers: {' | '.join(providers)}")

    # ------------------------------------------------------------------
    # Public API    
    # ------------------------------------------------------------------

    def call_structured_llm(
        self,
        prompt: str,
        system_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Execute an LLM prompt and parse the result into a Pydantic model.

        Automatically fails over through the provider chain:
        OpenRouter -> Groq -> Gemini.
        Raises an exception if all providers fail or are unconfigured.
        """
        any_configured = (
            self.openrouter_available
            or self.groq_available
            or self.gemini_available
        )
        if not any_configured:
            raise ValueError(
                f"[{self.agent_name}] No LLM provider configured. "
                "Please configure OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in your .env file."
            )

        # ── 1. OpenRouter (primary – free models) ─────────────────────────────────────
        if self.openrouter_available:
            try:
                return self._call_openrouter(prompt, system_prompt, response_schema)
            except _RateLimitError:
                print(f"[{self.agent_name}] OpenRouter rate-limited (quota) → trying Groq…")
            except _CreditExhaustedError as exc:
                print(f"[{self.agent_name}] OpenRouter quota exhausted ({str(exc)[:90]}) → disabled for this session")
            except Exception as exc:
                print(f"[{self.agent_name}] OpenRouter failed: {exc} → trying Groq…")

        # ── 2. Groq (secondary – fast free inference) ──────────────────────────────
        if self.groq_available:
            try:
                return self._call_groq(prompt, system_prompt, response_schema)
            except _RateLimitError:
                print(f"[{self.agent_name}] Groq rate-limited → trying Gemini…")
            except _CreditExhaustedError as exc:
                print(f"[{self.agent_name}] Groq quota exhausted ({str(exc)[:90]}) → disabled for this session")
            except Exception as exc:
                print(f"[{self.agent_name}] Groq failed: {exc} → trying Gemini…")

        # ── 3. Gemini (tertiary) ────────────────────────────────────────────────────
        if self.gemini_available:
            try:
                return self._call_gemini(prompt, system_prompt, response_schema)
            except Exception as exc:
                print(f"[{self.agent_name}] Gemini failed: {exc}")

        raise RuntimeError(
            f"[{self.agent_name}] All LLM providers exhausted or failed. "
            "Tried: "
            f"{', '.join(sorted(BaseAgent._tripped_providers)) or 'all configured providers'}. "
            "Free-tier quotas are exhausted – wait for reset or add credits at the provider console."
        )

    # ------------------------------------------------------------------
    # Generic OpenAI-compatible caller with retry/backoff
    # ------------------------------------------------------------------

    def _chat_completion_with_retry(
        self,
        provider: str,
        url: str,
        headers: dict,
        models: list,
        full_user_prompt: str,
        system_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Shared transport for all OpenAI-compatible providers (OpenRouter, Groq).

        - Walks the model list in priority order.
        - Retries transient failures (429 / 5xx) with exponential backoff.
        - On a schema-validation failure, issues one corrective repair call.
        """
        last_err: Optional[Exception] = None

        for model_id in models:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": full_user_prompt},
                ],
                "temperature": settings.TEMPERATURE,
                "max_tokens":  settings.MAX_TOKENS,
            }

            for attempt in range(1, settings.MAX_RETRIES_PER_MODEL + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=60)
                except requests.RequestException as exc:
                    last_err = exc
                    print(f"[{self.agent_name}] {provider} {model_id} network error ({exc}) → retry {attempt}/{settings.MAX_RETRIES_PER_MODEL}")
                    time.sleep(settings.RETRY_BACKOFF_SECONDS * attempt)
                    continue

                if resp.status_code == 200:
                    raw_text = (
                        resp.json().get("choices", [{}])[0]
                        .get("message", {})
                        .get("content")
                        or ""
                    )
                    if not raw_text.strip():
                        last_err = Exception(f"{provider} {model_id} returned empty content")
                        print(f"[{self.agent_name}] {provider} {model_id} empty content → trying next model…")
                        break

                    cleaned = self._extract_json_string(raw_text)
                    try:
                        result = response_schema.model_validate_json(cleaned)
                        print(f"[{self.agent_name}] ✅ {provider} success via {model_id}")
                        return result
                    except Exception as validation_exc:
                        last_err = validation_exc
                        print(f"[{self.agent_name}] {provider} {model_id} schema mismatch ({str(validation_exc)[:160]}) → attempting repair…")
                        repaired = self._repair_and_validate(
                            provider, url, headers, model_id, system_prompt,
                            raw_text, validation_exc, response_schema,
                        )
                        if repaired is not None:
                            print(f"[{self.agent_name}] ✅ {provider} success via {model_id} (repaired)")
                            return repaired
                        break

                elif resp.status_code == 429:
                    # Daily/monthly quota exhaustion is permanent for this
                    # session – trip the breaker instead of burning retries.
                    if _is_permanent_quota_exhaustion(resp.text):
                        self._trip_provider(provider)
                        raise _CreditExhaustedError(
                            f"{provider} quota permanently exhausted: {resp.text[:160]}"
                        )
                    last_err = _RateLimitError(f"{provider} {model_id} rate limited")
                    if attempt == settings.MAX_RETRIES_PER_MODEL:
                        print(f"[{self.agent_name}] {provider} {model_id} still rate-limited after {attempt} attempts → trying next model…")
                        break
                    wait = settings.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(f"[{self.agent_name}] {provider} {model_id} rate-limited → retry in {wait:.0f}s ({attempt}/{settings.MAX_RETRIES_PER_MODEL})")
                    time.sleep(wait)

                elif resp.status_code in (402, 403):
                    raise _CreditExhaustedError(f"{provider} billing/auth issue: {resp.text[:140]}")

                elif resp.status_code >= 500:
                    last_err = Exception(f"{provider} HTTP {resp.status_code} on {model_id}")
                    if attempt == settings.MAX_RETRIES_PER_MODEL:
                        print(f"[{self.agent_name}] {provider} {model_id} HTTP {resp.status_code} → trying next model…")
                        break
                    wait = settings.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(f"[{self.agent_name}] {provider} {model_id} HTTP {resp.status_code} → retry in {wait:.0f}s")
                    time.sleep(wait)

                else:
                    err_snippet = resp.text[:140]
                    last_err = Exception(f"{provider} HTTP {resp.status_code} on {model_id}: {err_snippet}")
                    print(f"[{self.agent_name}] {provider} {model_id} HTTP {resp.status_code} ({err_snippet}) → trying next model…")
                    break

        if last_err:
            raise last_err
        raise RuntimeError(f"{provider}: no models returned a valid response.")

    def _repair_and_validate(
        self,
        provider: str,
        url: str,
        headers: dict,
        model_id: str,
        system_prompt: str,
        bad_json: str,
        validation_exc: Exception,
        response_schema: Type[T],
    ) -> Optional[T]:
        """Ask the same model to fix its own schema violations (single repair attempt)."""
        repair_prompt = (
            f"Your previous response did not validate against the required schema.\n"
            f"VALIDATION ERROR:\n{str(validation_exc)[:800]}\n\n"
            f"YOUR PREVIOUS (INVALID) RESPONSE:\n{bad_json[:2500]}\n\n"
            f"Return the corrected, complete JSON object only. No prose, no markdown fences. "
            f"It must satisfy this schema:\n"
            f"{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )
        try:
            resp = requests.post(url, headers=headers, json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": repair_prompt},
                ],
                "temperature": 0.0,
                "max_tokens":  settings.MAX_TOKENS,
            }, timeout=60)
            if resp.status_code != 200:
                return None
            raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content") or ""
            if not raw.strip():
                return None
            return response_schema.model_validate_json(self._extract_json_string(raw))
        except Exception as exc:
            print(f"[{self.agent_name}] {provider} repair attempt failed: {str(exc)[:140]}")
            return None

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_openrouter(
        self,
        prompt: str,
        system_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Call OpenRouter using the OpenAI-compatible Chat Completions endpoint.
        Iterates through the free model list until one succeeds.
        """
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FAI-JobSearchAgent",
            "X-Title": "AI Job Search and Resume Tailoring Multi-Agent System",
        }

        full_user_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL INSTRUCTION: Your entire response MUST be a single, valid JSON object "
            f"that conforms exactly to this schema. Do NOT include any text, explanation, or "
            f"markdown outside the JSON object:\n"
            f"{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )

        return self._chat_completion_with_retry(
            "OpenRouter", settings.OPENROUTER_BASE_URL, headers,
            settings.OPENROUTER_FREE_MODELS, full_user_prompt, system_prompt, response_schema,
        )

    def _call_groq(
        self,
        prompt: str,
        system_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """
        Call Groq using its OpenAI-compatible Chat Completions endpoint.
        Iterates through available free Groq models until one succeeds.
        """
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        full_user_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL INSTRUCTION: Your entire response MUST be a single, valid JSON object "
            f"that conforms exactly to this schema. Do NOT include any text, explanation, or "
            f"markdown outside the JSON object:\n"
            f"{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )

        return self._chat_completion_with_retry(
            "Groq", settings.GROQ_BASE_URL, headers,
            settings.GROQ_MODELS, full_user_prompt, system_prompt, response_schema,
        )


    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        response_schema: Type[T],
    ) -> T:
        """Call Google Gemini using the google-genai SDK (v1 API)."""
        candidate_models = list(settings.GEMINI_MODELS)
        # Prepend user-specified model if set and not already listed
        if self.model_name and self.model_name not in candidate_models:
            candidate_models.insert(0, self.model_name)

        full_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL: Output must strictly conform to this JSON schema:\n"
            f"{json.dumps(response_schema.model_json_schema(), indent=2)}"
        )

        last_error: Optional[Exception] = None
        for m_name in candidate_models:
            for attempt in range(1, settings.MAX_RETRIES_PER_MODEL + 1):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=m_name,
                        contents=full_prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=settings.TEMPERATURE,
                            max_output_tokens=settings.MAX_TOKENS,
                            response_mime_type="application/json",
                        ),
                    )
                    raw_text = response.text or ""
                    cleaned  = self._extract_json_string(raw_text)
                    result   = response_schema.model_validate_json(cleaned)
                    print(f"[{self.agent_name}] ✅ Gemini success via {m_name}")
                    return result
                except Exception as exc:
                    msg = str(exc)
                    transient = ("429" in msg) or ("503" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("UNAVAILABLE" in msg)
                    last_error = exc
                    if transient and attempt < settings.MAX_RETRIES_PER_MODEL:
                        wait = settings.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                        print(f"[{self.agent_name}] Gemini '{m_name}' transient error → retry in {wait:.0f}s ({attempt}/{settings.MAX_RETRIES_PER_MODEL})")
                        time.sleep(wait)
                        continue
                    print(f"[{self.agent_name}] Gemini model '{m_name}' failed: {msg[:160]} → trying next…")
                    break

        if last_error:
            raise last_error

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """Extract and clean a JSON object from raw LLM output."""
        text = text.strip()

        # Strip ```json ... ``` fences
        if "```json" in text:
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                return m.group(1).strip()
        elif "```" in text:
            m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                return m.group(1).strip()

        # Grab the outermost { ... }
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        return text

"""
Agent 4: Match & Gap Analyzer Agent.
Academic Alignment: Module III (Heuristic Evaluation) & Module V (Inference & Logic).
Calculates multidimensional ATS compatibility and pinpoints critical skill deficiencies.
"""

import hashlib
import json
from typing import Optional

from config.settings import settings
from schemas.resume_schema import CandidateProfile
from schemas.job_schema import JobListing
from schemas.analysis_schema import MatchReport, MatchBreakdown
from agents.base_agent import BaseAgent
from tools.job_cache import JobCache


class MatchGapAgent(BaseAgent):
    """
    Evaluator agent that measures candidate-to-job compatibility,
    identifies missing keywords, and formulates specific tailoring strategies.

    Verdicts are memoised in SQLite. Scoring one job is the most expensive
    repeated call in the pipeline (the UI scores five jobs per search), so a
    previously computed verdict for the same candidate + job + model is
    returned straight from the cache instead of being re-generated.
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(agent_name="MatchGapAgent", model_name=model_name)
        self._cache = JobCache(settings.SQLITE_DB_PATH)

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------
    def _active_model_id(self) -> str:
        """Best-effort identifier for the model that will answer this call."""
        if self.openrouter_available:
            return settings.OPENROUTER_FREE_MODELS[0]
        if self.groq_available:
            return settings.GROQ_MODELS[0]
        return self.model_name or settings.DEFAULT_LLM_MODEL

    def _profile_hash(self, profile: CandidateProfile) -> str:
        """
        Fingerprint everything that can change a verdict.

        Folding the active model into the key means switching the model list
        invalidates old scores rather than silently serving verdicts produced
        under different reasoning.
        """
        payload = json.dumps(
            {
                "skills": sorted(profile.skills.all_technical_skills()),
                "projects": sorted(
                    p.title + "|" + ",".join(sorted(p.technologies))
                    for p in profile.projects
                ),
                "experience": profile.total_years_experience,
                "seniority": profile.inferred_seniority_level,
                "model": self._active_model_id(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def analyze_match(self, profile: CandidateProfile, job: JobListing) -> MatchReport:
        """
        Produce a comprehensive MatchReport for a specific job opening.

        Served from the match-report cache when an identical candidate, job and
        model combination has already been scored.
        """
        cache_key = self._profile_hash(profile)

        cached_json = self._cache.get_match_report(cache_key, job.id)
        if cached_json:
            try:
                report = MatchReport.model_validate_json(cached_json)
                print(
                    f"[{self.agent_name}] cache hit for job {job.id} "
                    f"(score {report.overall_match_score}) - skipping LLM"
                )
                return report
            except Exception as exc:
                # A stale or corrupt row must never break the pipeline.
                print(
                    f"[{self.agent_name}] discarding unreadable cached report "
                    f"for {job.id}: {str(exc)[:120]}"
                )

        system_prompt = (
            "You are an expert ATS Match & Gap Evaluation Agent. "
            "Your objective is to compare a candidate's verified profile against a job description. "
            "SCORING FORMULA: "
            "- Hard Skills Match (50% weight): Exact or conceptual match of candidate's technical skills vs JD. "
            "- Project Relevance (30% weight): How well candidate's past projects align with JD domain. "
            "- Experience/Education (20% weight): Appropriate seniority and educational background. "
            "RULES: "
            "1. Be strictly objective and realistic. Do not give 95%+ unless the candidate is an exact match. "
            "2. Identify explicitly which skills are MATCHED and which are MISSING. "
            "3. Provide actionable tailoring recommendations without suggesting fabrication."
        )

        user_prompt = (
            f"Candidate Skills: {', '.join(profile.skills.all_technical_skills())}\n"
            f"Candidate Projects: {'; '.join([p.title + ': ' + ', '.join(p.technologies) for p in profile.projects])}\n"
            f"Candidate Experience: {profile.total_years_experience} yrs ({profile.inferred_seniority_level})\n\n"
            f"Target Job: {job.title} at {job.company}\n"
            f"Required Skills: {', '.join(job.required_skills)}\n"
            f"Nice-To-Have Skills: {', '.join(job.nice_to_have_skills)}\n"
            f"Job Description Excerpt:\n{job.description[:1200]}\n\n"
            f"Calculate the ATS match and gap report conforming to MatchReport schema."
        )

        report = self.call_structured_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=MatchReport,
        )

        report.job_id = job.id
        report.job_title = job.title
        report.company = job.company

        # Cache failures are non-fatal; the report is still returned.
        try:
            self._cache.save_match_report(
                cache_key,
                job.id,
                report.model_dump_json(),
                model_id=self._active_model_id(),
            )
        except Exception as exc:
            print(f"[{self.agent_name}] could not cache report: {str(exc)[:120]}")

        return report

    def _calculate_heuristic_match(self, profile: CandidateProfile, job: JobListing) -> MatchReport:
        """
        Deterministic heuristic evaluation function (Academic Module III).
        Computes mathematical set intersection of skills and weighted scoring.
        """
        candidate_skills = {s.lower().strip() for s in profile.skills.all_technical_skills()}
        required_skills = [s.strip() for s in job.required_skills] or ["Python"]

        matched = []
        missing = []

        for req in required_skills:
            req_clean = req.lower()
            if any(req_clean in cs or cs in req_clean for cs in candidate_skills):
                matched.append(req)
            else:
                missing.append(req)

        # Calculate component scores
        skill_ratio = len(matched) / max(len(required_skills), 1)
        hard_skills_score = int(skill_ratio * 100)

        # Project score heuristic based on keyword overlap
        project_text = " ".join([p.title + " " + " ".join(p.technologies) for p in profile.projects]).lower()
        project_overlap = sum(1 for req in required_skills if req.lower() in project_text)
        project_score = min(int((project_overlap / max(len(required_skills), 1)) * 100) + 15, 95)

        # Experience score
        exp_score = 85 if "intern" in job.experience_level.lower() or "entry" in job.experience_level.lower() else 70

        # Weighted aggregate: 50% skills + 30% projects + 20% experience
        overall_score = int((hard_skills_score * 0.50) + (project_score * 0.30) + (exp_score * 0.20))

        breakdown = MatchBreakdown(
            hard_skills_score=hard_skills_score,
            project_relevance_score=project_score,
            experience_education_score=exp_score
        )

        recommendations = [
            f"Prominently highlight {', '.join(matched[:3])} in your summary and project bullets.",
            f"If you have basic exposure to {', '.join(missing[:2]) or 'the missing requirements'}, mention relevant coursework or GitHub explorations.",
            "Rephrase project action verbs to emphasize quantifiable performance improvements."
        ]

        return MatchReport(
            job_id=job.id,
            job_title=job.title,
            company=job.company,
            overall_match_score=overall_score,
            score_breakdown=breakdown,
            matched_skills=matched,
            missing_skills=missing,
            strengths_summary=f"Strong match in core foundations: {', '.join(matched[:4]) or 'general software engineering'}.",
            tailoring_recommendations=recommendations
        )

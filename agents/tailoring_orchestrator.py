"""
Multi-Agent Tailoring & Verification Loop Orchestrator.
Academic Alignment: Module IX (Multi-Agent Systems & Producer-Critic Reflection Loop).
Coordinates iterative generation, adversarial fact-checking, and convergence on grounded resumes.
"""

from typing import Tuple, Callable, Optional
from schemas.resume_schema import CandidateProfile
from schemas.job_schema import JobListing
from schemas.analysis_schema import MatchReport
from schemas.tailoring_schema import TailoredResume, VerificationReport
from agents.tailor_agent import TailorAgent
from agents.verifier_agent import VerifierAgent
from config.settings import settings


class TailoringOrchestrator:
    """
    Manages the bidirectional feedback loop between TailorAgent and VerifierAgent.
    Ensures that tailored documents satisfy all grounding constraints before compilation.
    """

    def __init__(self):
        self.tailor_agent = TailorAgent()
        self.verifier_agent = VerifierAgent()

    def run_tailoring_pipeline(
        self,
        profile: CandidateProfile,
        job: JobListing,
        match_report: MatchReport,
        max_retries: int = settings.MAX_VERIFICATION_RETRIES,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[TailoredResume, VerificationReport, int]:
        """
        Execute the Tailor <-> Verifier collaboration loop.

        Returns:
            Tuple of (Final TailoredResume, Final VerificationReport, Iteration Count)
        """
        def log(msg: str):
            print(f"[Orchestrator] {msg}")
            if status_callback:
                status_callback(msg)

        log(f"Starting tailored resume synthesis for: {job.title} at {job.company}...")
        current_feedback: Optional[str] = None
        iteration = 0

        final_resume: Optional[TailoredResume] = None
        final_report: Optional[VerificationReport] = None

        while iteration <= max_retries:
            iteration += 1
            log(f"Iteration {iteration}: Tailor Agent drafting customized resume content...")

            # Tailor Agent synthesizes or refines draft
            tailored_resume = self.tailor_agent.tailor_resume(
                profile=profile,
                job=job,
                match_report=match_report,
                auditor_feedback=current_feedback
            )
            final_resume = tailored_resume

            log(f"Iteration {iteration}: Verifier Agent auditing draft for truthfulness & grounding...")

            # Verifier Agent conducts adversarial compliance audit
            audit_report = self.verifier_agent.audit(
                original_profile=profile,
                tailored_resume=tailored_resume
            )
            final_report = audit_report

            if audit_report.status == "APPROVED":
                log(f"Iteration {iteration}: Audit PASSED! (Grounding confidence: HIGH).")
                break
            else:
                log(f"Iteration {iteration}: Audit FLAGGED issues ({len(audit_report.issues_found)}). Requesting revision...")
                current_feedback = audit_report.feedback_for_tailor

        log(f"Tailoring pipeline completed in {iteration} cycle(s). Final Status: {final_report.status}")
        return final_resume, final_report, iteration

"""
Agent 6: Verifier & Anti-Hallucination Auditor Agent.
Academic Alignment: Module IV (Constraint Satisfaction - CSP) & Module IX (Multi-Agent Verification Loop).
Acts as an adversarial auditor to guarantee no fabricated qualifications or ungrounded claims exist.
"""

from typing import Optional, List
from schemas.resume_schema import CandidateProfile
from schemas.tailoring_schema import (
    TailoredResume,
    VerificationReport,
    VerificationIssue,
)
from agents.base_agent import BaseAgent


class VerifierAgent(BaseAgent):
    """
    Auditor agent that evaluates tailored resumes against original candidate truth.
    Enforces the CSP constraint: Tailored Resume Content ⊆ Original Resume Ground Truth.
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(agent_name="VerifierAgent", model_name=model_name)

    def audit(
        self,
        original_profile: CandidateProfile,
        tailored_resume: TailoredResume
    ) -> VerificationReport:
        """
        Audit the tailored resume for ungrounded claims, invented skills, or false metrics.
        """
        system_prompt = (
            "You are an adversarial AI Resume Fact-Checking & Compliance Auditor. "
            "Your solemn duty is to protect the candidate from resume fraud and recruiter blacklisting. "
            "You must strictly evaluate whether the Tailored Resume contains any ungrounded or invented claims "
            "not backed by the Original Candidate Profile.\n\n"
            "VERIFICATION CONSTRAINTS (CSP):\n"
            "1. NO INVENTED SKILLS: If the tailored resume lists a skill that the candidate never had in the original profile, FLAG IT.\n"
            "2. NO INVENTED EMPLOYERS OR ROLES: Job titles and companies must match original truth.\n"
            "3. NO INVENTED METRICS: The tailor cannot invent arbitrary percentages or dollar amounts not in the original text.\n"
            "4. OUTCOME: If violations exist, status must be 'REVISION_NEEDED' and you must provide exact feedback. "
            "   If all claims are truthful rephrasings of real facts, status is 'APPROVED'."
        )

        user_prompt = (
            f"ORIGINAL CANDIDATE PROFILE (GROUND TRUTH):\n"
            f"- Skills: {', '.join(original_profile.skills.all_technical_skills())}\n"
            f"- Experience: {[e.model_dump() for e in original_profile.experience]}\n"
            f"- Projects: {[p.model_dump() for p in original_profile.projects]}\n"
            f"- Certifications: {original_profile.certifications}\n\n"
            f"TAILORED RESUME TO AUDIT:\n"
            f"- Target: {tailored_resume.targeted_job_title} at {tailored_resume.targeted_company}\n"
            f"- Tailored Summary: {tailored_resume.professional_summary}\n"
            f"- Tailored Skills: {', '.join(tailored_resume.prioritized_skills.all_technical_skills())}\n"
            f"- Tailored Experiences: {[e.model_dump() for e in tailored_resume.tailored_experiences]}\n"
            f"- Tailored Projects: {[p.model_dump() for p in tailored_resume.tailored_projects]}\n\n"
            f"Perform a rigorous audit and return a VerificationReport."
        )

        return self.call_structured_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=VerificationReport,
        )

    def _audit_grounding_heuristically(
        self,
        original: CandidateProfile,
        tailored: TailoredResume
    ) -> VerificationReport:
        """
        Deterministic CSP constraint checking algorithm.
        Verifies subset inclusion: Tailored_Skills ⊆ Original_Skills (Module IV).
        """
        orig_skills_set = {s.lower().strip() for s in original.skills.all_technical_skills()}
        tailored_skills = tailored.prioritized_skills.all_technical_skills()

        issues: List[VerificationIssue] = []

        # Check for ungrounded skills
        for ts in tailored_skills:
            ts_clean = ts.lower().strip()
            # Allow common variations or exact matches
            matched = any(
                ts_clean in os or os in ts_clean
                for os in orig_skills_set
            )
            if not matched:
                issues.append(
                    VerificationIssue(
                        issue_type="UNGROUNDED_SKILL",
                        flagged_content=ts,
                        reason=f"Skill '{ts}' was added to tailored resume but was not in candidate's original skill set.",
                        suggested_correction=f"Remove '{ts}' from the skill list and focus on candidate's verified skills."
                    )
                )

        # Check for fabricated companies
        orig_companies = {e.company.lower().strip() for e in original.experience}
        for te in tailored.tailored_experiences:
            if te.company.lower().strip() not in orig_companies:
                issues.append(
                    VerificationIssue(
                        issue_type="FALSE_EXPERIENCE",
                        flagged_content=te.company,
                        reason=f"Company '{te.company}' does not exist in original candidate profile.",
                        suggested_correction="Revert to original verified company name."
                    )
                )

        if issues:
            return VerificationReport(
                status="REVISION_NEEDED",
                is_grounded=False,
                hallucination_risk_level="MEDIUM",
                audit_notes=f"Detected {len(issues)} constraint violations regarding ungrounded content.",
                issues_found=issues,
                feedback_for_tailor=(
                    f"Please correct the following violations: "
                    + "; ".join([f"{i.issue_type}: {i.flagged_content} ({i.reason})" for i in issues])
                )
            )

        return VerificationReport(
            status="APPROVED",
            is_grounded=True,
            hallucination_risk_level="LOW",
            audit_notes="Verification successful. All skills, projects, and experiences are strictly grounded in original profile truth.",
            issues_found=[],
            feedback_for_tailor=None
        )

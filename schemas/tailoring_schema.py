"""
Pydantic Data Models for Tailored Resumes and Grounding Verification.
Academic Alignment: Module IV (Constraint Satisfaction) & Module IX (Multi-Agent Systems).
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from .resume_schema import CandidateContact, EducationItem, TechnicalSkills


class TailoredBullet(BaseModel):
    """Refactored bullet point tailored for a specific JD."""
    original_text: str = Field(..., description="Original text from candidate's resume")
    tailored_text: str = Field(..., description="Rephrased bullet emphasizing JD-relevant keywords truthfully")
    keywords_highlighted: List[str] = Field(default_factory=list, description="Target keywords integrated into bullet")


class TailoredExperience(BaseModel):
    """Work experience with aligned bullet points."""
    company: str = Field(..., description="Company name")
    role: str = Field(..., description="Role title")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    location: Optional[str] = Field(None, description="Location")
    tailored_bullets: List[TailoredBullet] = Field(
        default_factory=list,
        description="Refactored experience bullets"
    )
    technologies_used: List[str] = Field(default_factory=list)


class TailoredProject(BaseModel):
    """Project with aligned bullet points."""
    title: str = Field(..., description="Project title")
    technologies: List[str] = Field(default_factory=list, description="Technologies prioritized for this JD")
    tailored_bullets: List[TailoredBullet] = Field(
        default_factory=list,
        description="Refactored project bullets"
    )
    github_or_live_link: Optional[str] = Field(None)


class TailoredResume(BaseModel):
    """
    Complete, validated tailored resume data structure ready for PDF compilation.
    Synthesized by Agent 5 (Tailor Agent) and verified by Agent 6 (Auditor).
    """
    candidate_name: str = Field(..., description="Candidate full name")
    contact: CandidateContact = Field(..., description="Contact details")
    targeted_job_title: str = Field(..., description="Job title being applied to")
    targeted_company: str = Field(..., description="Target company name")
    professional_summary: str = Field(
        ...,
        description="Tailored professional summary aligned with the target JD while remaining 100% truthful"
    )
    prioritized_skills: TechnicalSkills = Field(
        ...,
        description="Re-ordered and prioritized skills matching the JD requirements"
    )
    tailored_experiences: List[TailoredExperience] = Field(
        default_factory=list,
        description="Work experiences with tailored bullets"
    )
    tailored_projects: List[TailoredProject] = Field(
        default_factory=list,
        description="Projects with tailored bullets"
    )
    education: List[EducationItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    ats_score_before: int = Field(default=0, description="ATS match score of original resume")
    ats_score_after: int = Field(default=0, description="Projected ATS match score of tailored resume")


class VerificationIssue(BaseModel):
    """A detected violation of truthfulness or grounding constraints."""
    issue_type: str = Field(
        ...,
        description="e.g., 'HALLUCINATED_SKILL', 'INVENTED_METRIC', 'FALSE_EXPERIENCE', 'TITLE_EXAGGERATION'"
    )
    flagged_content: str = Field(..., description="Snippet from tailored resume that violated constraints")
    reason: str = Field(..., description="Why this content is considered ungrounded or prohibited")
    suggested_correction: str = Field(..., description="How to rectify the issue without inventing data")


class VerificationReport(BaseModel):
    """
    Auditor result produced by Agent 6 (Verifier Agent).
    Enforces the CSP constraint: Tailored Resume Content ⊆ Original Resume Content.
    """
    status: str = Field(
        ...,
        description="'APPROVED' if 100% grounded and compliant; 'REVISION_NEEDED' if violations found"
    )
    is_grounded: bool = Field(..., description="True if no fabricated claims or ungrounded skills were introduced")
    hallucination_risk_level: str = Field(default="LOW", description="'LOW', 'MEDIUM', or 'HIGH'")
    audit_notes: str = Field(..., description="Executive summary of the verification inspection")
    issues_found: List[VerificationIssue] = Field(default_factory=list, description="List of detected discrepancies")
    feedback_for_tailor: Optional[str] = Field(
        None,
        description="Actionable instruction to send back to Tailor Agent if REVISION_NEEDED"
    )

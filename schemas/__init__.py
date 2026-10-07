"""Package initialization for all Pydantic schemas."""
from .resume_schema import (
    CandidateContact,
    EducationItem,
    WorkExperienceItem,
    ProjectItem,
    TechnicalSkills,
    CandidateProfile,
)
from .job_schema import (
    JobListing,
    JobSearchQuery,
    RoleRecommendation,
    RoleRecommendationResponse,
)
from .analysis_schema import (
    SkillMatchItem,
    MatchBreakdown,
    MatchReport,
)
from .tailoring_schema import (
    TailoredBullet,
    TailoredExperience,
    TailoredProject,
    TailoredResume,
    VerificationIssue,
    VerificationReport,
)

__all__ = [
    "CandidateContact",
    "EducationItem",
    "WorkExperienceItem",
    "ProjectItem",
    "TechnicalSkills",
    "CandidateProfile",
    "JobListing",
    "JobSearchQuery",
    "RoleRecommendation",
    "RoleRecommendationResponse",
    "SkillMatchItem",
    "MatchBreakdown",
    "MatchReport",
    "TailoredBullet",
    "TailoredExperience",
    "TailoredProject",
    "TailoredResume",
    "VerificationIssue",
    "VerificationReport",
]

"""
Pydantic Data Models for the Candidate Profile and Resume Representation.
Academic Alignment: Module VI (Knowledge Representation & Ontological Engineering).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CandidateContact(BaseModel):
    """Candidate contact information."""
    name: str = Field(..., description="Full name of the candidate")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number if available")
    location: Optional[str] = Field(None, description="City, State / Country")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio_url: Optional[str] = Field(None, description="Personal portfolio website")


class EducationItem(BaseModel):
    """Academic credential item."""
    institution: str = Field(..., description="Name of college/university")
    degree: str = Field(..., description="Degree (e.g., B.Tech in Computer Science, B.S.)")
    field_of_study: Optional[str] = Field(None, description="Major / Specialization")
    start_year: Optional[str] = Field(None, description="Start year")
    end_year: Optional[str] = Field(None, description="Graduation year / Expected graduation")
    grade_or_gpa: Optional[str] = Field(None, description="GPA or percentage, e.g. '8.6/10' or '3.8/4.0'")


class WorkExperienceItem(BaseModel):
    """Past work or internship experience item."""
    company: str = Field(..., description="Company or organization name")
    role: str = Field(..., description="Job title, e.g. 'Machine Learning Intern'")
    location: Optional[str] = Field(None, description="Job location or 'Remote'")
    start_date: Optional[str] = Field(None, description="Start date (Month Year)")
    end_date: Optional[str] = Field(None, description="End date (Month Year or 'Present')")
    is_current: bool = Field(default=False, description="Whether this is the candidate's current role")
    description_bullets: List[str] = Field(
        default_factory=list,
        description="Factual achievement bullet points describing tasks and impact"
    )
    technologies_used: List[str] = Field(
        default_factory=list,
        description="Specific tools and technologies used in this role"
    )


class ProjectItem(BaseModel):
    """Portfolio or academic project item."""
    title: str = Field(..., description="Name of the project")
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies, frameworks, and tools used in this project"
    )
    description_bullets: List[str] = Field(
        default_factory=list,
        description="Bullet points describing the problem, solution, and measurable results"
    )
    github_or_live_link: Optional[str] = Field(None, description="URL to repository or demo")


class TechnicalSkills(BaseModel):
    """Categorized technical capabilities."""
    languages: List[str] = Field(default_factory=list, description="Programming languages (e.g., Python, C++, Java, JavaScript)")
    frameworks_libraries: List[str] = Field(default_factory=list, description="Frameworks (e.g., PyTorch, TensorFlow, FastAPI, React, LangChain)")
    databases: List[str] = Field(default_factory=list, description="Databases (e.g., PostgreSQL, MongoDB, Pinecone, Redis)")
    cloud_devops: List[str] = Field(default_factory=list, description="Cloud & DevOps tools (e.g., Docker, AWS, Git, GitHub Actions)")
    developer_tools: List[str] = Field(default_factory=list, description="Developer utilities (e.g., VS Code, Postman, Linux)")
    soft_skills: List[str] = Field(default_factory=list, description="Interpersonal & teamwork capabilities")

    def all_technical_skills(self) -> List[str]:
        """Aggregate all distinct technical skill keywords."""
        all_skills = (
            self.languages +
            self.frameworks_libraries +
            self.databases +
            self.cloud_devops +
            self.developer_tools
        )
        # Deduplicate while preserving order (case-insensitive deduplication)
        seen = set()
        deduped = []
        for s in all_skills:
            clean = s.strip()
            if clean and clean.lower() not in seen:
                seen.add(clean.lower())
                deduped.append(clean)
        return deduped


class CandidateProfile(BaseModel):
    """
    Comprehensive Ontological Knowledge Representation of the Candidate.
    Serves as the Single Source of Truth across all downstream agents.
    """
    contact: CandidateContact = Field(..., description="Contact details")
    professional_summary: str = Field(..., description="Concise professional summary")
    education: List[EducationItem] = Field(default_factory=list, description="Academic background")
    experience: List[WorkExperienceItem] = Field(default_factory=list, description="Internships or full-time roles")
    projects: List[ProjectItem] = Field(default_factory=list, description="Key software or AI projects")
    skills: TechnicalSkills = Field(default_factory=TechnicalSkills, description="Categorized skill matrix")
    certifications: List[str] = Field(default_factory=list, description="Official licenses or certifications")
    total_years_experience: float = Field(default=0.0, description="Estimated total experience in years")
    inferred_seniority_level: str = Field(
        default="Fresher / Entry-Level",
        description="e.g., Fresher / Entry-Level (0-1 yrs), Junior (1-3 yrs), Mid-Level (3-5 yrs)"
    )
    raw_text: Optional[str] = Field(None, description="Original parsed resume text for ground truth validation")

    def all_skills_flat(self) -> List[str]:
        """Convenience accessor to flatten and return all technical skills."""
        if hasattr(self, "skills") and self.skills:
            return self.skills.all_technical_skills()
        return []

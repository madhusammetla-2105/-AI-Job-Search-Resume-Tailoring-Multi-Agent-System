"""
Agent 2: Career Strategy & Role Recommender.
Academic Alignment: Module VII (State Space Planning & Goal Formulation).
Analyzes the CandidateProfile ontology and recommends top 3-5 high-fit roles with rationale.
"""

from typing import Optional
from schemas.resume_schema import CandidateProfile
from schemas.job_schema import (
    RoleRecommendation,
    RoleRecommendationResponse,
)
from agents.base_agent import BaseAgent


class RoleRecommenderAgent(BaseAgent):
    """
    Agent responsible for analyzing candidate capabilities and defining
    the target state space (the top job roles to pursue).
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(agent_name="RoleRecommenderAgent", model_name=model_name)

    def recommend_roles(self, profile: CandidateProfile) -> RoleRecommendationResponse:
        """
        Synthesize candidate profile to recommend top 3-5 roles.
        """
        system_prompt = (
            "You are an expert AI Career Strategist & State Space Planning Agent. "
            "Given a candidate's structured profile (skills, projects, experience, education), "
            "formulate the optimal goal states by recommending 3 to 5 highly realistic, high-fit "
            "job roles suitable for their seniority level. "
            "GUIDELINES: "
            "1. Match roles realistically to their seniority (e.g. recommend Trainee, Intern, Junior, or Associate roles for freshers). "
            "2. Provide an honest fit_score (0-100) reflecting their genuine alignment. "
            "3. State a crisp, logical rationale based on their actual projects and skills. "
            "4. Provide search keywords optimized for job portals (e.g. 'Junior Machine Learning Engineer', 'Python Backend Developer')."
        )

        user_prompt = (
            f"Candidate Overview:\n"
            f"- Name: {profile.contact.name}\n"
            f"- Seniority Level: {profile.inferred_seniority_level}\n"
            f"- Experience: {profile.total_years_experience} years\n"
            f"- Skills: {', '.join(profile.skills.all_technical_skills()[:25])}\n"
            f"- Summary: {profile.professional_summary}\n"
            f"- Projects: {', '.join([p.title for p in profile.projects])}\n\n"
            f"Analyze this profile and recommend the top 3-5 job roles."
        )

        return self.call_structured_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_schema=RoleRecommendationResponse,
        )

    def _build_deterministic_recommendations(self, profile: CandidateProfile) -> RoleRecommendationResponse:
        """Fallback simulation generator driven dynamically by candidate's parsed skills across 4 diverse roles."""
        all_skills = profile.all_skills_flat()
        skills_str = ", ".join(all_skills[:5]) if all_skills else "Software Engineering"
        is_fresher = profile.total_years_experience < 1.0

        # Skill domain detection
        is_data_ml = any(s.lower() in ["machine learning", "pytorch", "tensorflow", "pandas", "data science", "nlp", "ai", "scikit-learn"] for s in all_skills)
        is_frontend = any(s.lower() in ["react", "javascript", "typescript", "html", "css", "vue", "angular", "frontend", "next.js"] for s in all_skills)
        is_backend = any(s.lower() in ["python", "fastapi", "django", "flask", "java", "sql", "postgresql", "node.js", "express", "c++"] for s in all_skills)
        is_cloud = any(s.lower() in ["docker", "kubernetes", "aws", "gcp", "azure", "linux", "ci/cd", "git"] for s in all_skills)

        roles = []

        # Role 1: Primary Core Match
        if is_data_ml:
            r1_title = "Junior Machine Learning / AI Engineer" if is_fresher else "Machine Learning Engineer"
            r1_score = 92
            r1_rat = f"Strong grounding in statistical models and ML frameworks: {', '.join([s for s in all_skills if s.lower() in ['python', 'pytorch', 'tensorflow', 'pandas', 'scikit-learn']][:4])}."
        elif is_frontend:
            r1_title = "Junior Frontend / Web Developer" if is_fresher else "Frontend Developer"
            r1_score = 91
            r1_rat = f"Direct alignment with web client technologies: {', '.join([s for s in all_skills if s.lower() in ['react', 'javascript', 'typescript', 'html', 'css']][:4])}."
        else:
            r1_title = "Junior Python / Backend Developer" if is_fresher else "Backend Software Engineer"
            r1_score = 90
            r1_rat = f"Demonstrated backend programming capabilities: {', '.join([s for s in all_skills if s.lower() in ['python', 'fastapi', 'django', 'sql', 'java']][:4])}."
        
        roles.append(RoleRecommendation(
            role_title=r1_title,
            fit_score=r1_score,
            rationale=r1_rat,
            key_matching_strengths=all_skills[:5],
            recommended_search_keywords=[r1_title, f"{r1_title} Fresher"]
        ))

        # Role 2: Full-Stack / Software Development Engineer (SDE)
        r2_title = "Associate Software Development Engineer (SDE)" if is_fresher else "Software Development Engineer"
        roles.append(RoleRecommendation(
            role_title=r2_title,
            fit_score=87,
            rationale=f"Comprehensive core computer science foundations and project development in {skills_str}.",
            key_matching_strengths=all_skills[:4],
            recommended_search_keywords=[r2_title, "Software Engineer Entry Level"]
        ))

        # Role 3: Specialized Domain Track (Data Analytics or API Development)
        if is_data_ml or any("sql" in s.lower() or "python" in s.lower() for s in all_skills):
            r3_title = "Data Analyst / Analytics Associate"
            r3_rat = "Proficiency in query formulation, tabular data handling, and analytical logic."
            r3_strengths = [s for s in all_skills if s.lower() in ["sql", "python", "pandas", "excel"]] or all_skills[:3]
        else:
            r3_title = "API & Integration Engineer"
            r3_rat = "Ability to connect microservices, databases, and structured HTTP REST endpoints."
            r3_strengths = all_skills[:3]

        roles.append(RoleRecommendation(
            role_title=r3_title,
            fit_score=84,
            rationale=r3_rat,
            key_matching_strengths=r3_strengths,
            recommended_search_keywords=[r3_title, "Associate Analyst"]
        ))

        # Role 4: Modern AI / Cloud / DevOps Trainee
        if is_cloud:
            r4_title = "Cloud & DevOps Trainee Engineer"
            r4_rat = "Practical exposure to containerization, version control, and Linux environments."
        else:
            r4_title = "Applied Generative AI & Automation Intern"
            r4_rat = "Strong curiosity and application-level development in intelligent workflows."

        roles.append(RoleRecommendation(
            role_title=r4_title,
            fit_score=81,
            rationale=r4_rat,
            key_matching_strengths=all_skills[:4],
            recommended_search_keywords=[r4_title, "AI Intern"]
        ))

        return RoleRecommendationResponse(
            candidate_summary_analysis=(
                f"Candidate {profile.contact.name} ({profile.inferred_seniority_level}) presents verifiable competencies in {skills_str}. "
                f"Formulated 4 targeted career pathways spanning core software engineering, domain specialization, and applied emerging technologies."
            ),
            recommended_roles=roles
        )

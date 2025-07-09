from pydantic import BaseModel, Field
from typing import Literal, Optional

Position = Literal["FE", "BE", "FS", "DEV_OPS", "DATA_SCIENTIST", "DATA_ENGINEER" ,"AI_ENGINEER", "OTHER"]

class CandidateEducation(BaseModel):
    institution: str = Field(description="The institution of the candidate")
    degree: Optional[str] = Field(description="The degree of the candidate")
    field_of_study: Optional[str] = Field(description="The field of study of the candidate")
    start_date: Optional[str] = Field(description="The start date of the candidate's education. YYYY-MM format.")
    end_date: Optional[str] = Field(description="The end date of the candidate's education. YYYY-MM format.")
    description: str = Field(description="The description of the candidate's education")

class CandidateProfile(BaseModel):
    name: str = Field(description="The name of the candidate")
    position: Position = Field(description="The desired developer position the candidate is applying for. Infer this from the resume's objective, title, or summary. Must be one of the values from the Position literal.")
    objective: str = Field(description="The objective of the candidate.")
    experience_years: Optional[Literal["JUNIOR", "SENIOR", "STAFF", "PRINCIPAL"]] = Field(default=None, description="The experience level of the candidate. JUNIOR: 0-3 years, SENIOR: 4-8 years, STAFF: 9-12 years, PRINCIPAL: 13+ years.")
    education: list[CandidateEducation] = Field(default=[], description="The education of the candidate")

class BaseExperience(BaseModel):
    """Base class for all types of experiences - contains common fields shared between career and project experiences"""
    start_date: Optional[str] = Field(description="The start date of the experience in YYYY-MM format.")
    end_date: Optional[str] = Field(description="The end date of the experience in YYYY-MM format.")
    tech_stack: list[str] = Field(description="List of technologies, programming languages, frameworks, tools, and platforms used during this experience. Be specific (e.g., 'React 18', 'Node.js', 'PostgreSQL', 'AWS EC2').")
    architecture: Optional[str] = Field(default=None, description="Detailed description of system architecture, design patterns, or technical infrastructure if relevant. Use 'mermaid' format to describe the architecture. Leave empty if not applicable.")
    position: list[Position] = Field(default=[], description="List of developer roles performed by the candidate during this specific experience (company or project). Focus on the actual development role (e.g., 'FE', 'BE'), not generic titles like 'Team Leader' or 'Part Manager'. Must use values from the Position enum. If multiple roles were held, list all relevant ones.")
    summary: str = Field(description="A comprehensive summary of the experience including key responsibilities, achievements, and overall impact. Should be 2-3 sentences.")
    # STAR Method fields
    situation: list[str] = Field(default=[], description="STAR method - Situation: List of specific situations or contexts that required action. Each item should describe a challenge, problem, or scenario faced.")
    task: list[str] = Field(default=[], description="STAR method - Task: List of specific tasks, responsibilities, or objectives that needed to be accomplished in each situation.")
    action: list[str] = Field(default=[], description="STAR method - Action: List of specific actions taken to address the tasks. Focus on individual contributions and methodologies used.")
    result: list[str] = Field(default=[], description="STAR method - Result: List of quantifiable outcomes, achievements, or impacts. Include metrics, improvements, or success indicators whenever possible.")


class CareerExperience(BaseExperience):
    company: str = Field(description="The name of the company or organization where the candidate worked. This should be an official employer, including full-time jobs, part-time positions, internships, contract work, or freelance work for specific companies.")
    company_description: str = Field(description="A brief description of the company or organization where the candidate worked. This should be a short summary of the company's business, industry, and key products or services.")
    employee_type: Literal["EMPLOYEE", "INTERN", "CONTRACT", "FREELANCE"] = Field(description="Type of employee: EMPLOYEE (full-time employee), INTERN (internship), CONTRACT (contract work), FREELANCE (freelance work)")
    job_level: Optional[str] = Field(default=None, description="Job level or rank at the company. Leave empty if not specified or unclear.")    
    

class ProjectExperience(BaseExperience):
    project_name: str = Field(description="The name or title of the project. This should be independent work, personal projects, side projects, open source contributions, hackathon projects, or academic projects - NOT projects done as part of regular employment.")
    project_type: Literal["PERSONAL", "TEAM", "OPEN_SOURCE", "ACADEMIC", "HACKATHON", "FREELANCE"] = Field(description="Type of project: PERSONAL (solo project), TEAM (collaborative project), OPEN_SOURCE (contribution to open source), ACADEMIC (school/research project), HACKATHON (competition project), FREELANCE (independent client work)")
    team_size: Optional[int] = Field(default=None, description="Number of people involved in the project including the candidate. Use 1 for solo projects, or the total team size for collaborative projects.")

class ResumeParseResult(BaseModel):
    candidate_profile: CandidateProfile = Field(description="The candidate profile of the candidate")
    career_experiences: list[CareerExperience] = Field(default=[], description="Professional work experiences at companies including full-time, part-time, contract, and internship positions")
    project_experiences: list[ProjectExperience] = Field(default=[], description="Independent project experience including personal projects, team projects, open source contributions, and academic projects that are NOT part of regular employment")


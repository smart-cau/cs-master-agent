from __future__ import annotations
from langchain_core.documents import Document

from parsing_graph.schema.schema import (
    ResumeParseResult,
    CandidateProfile,
    CareerExperience,
    ProjectExperience,
)


def _convert_profile_to_document(
    profile: CandidateProfile
) -> Document:
    """Converts a CandidateProfile to a Document."""
    page_content = f"""
# Candidate Profile: {profile.name}

## Position
- **Desired Position:** {profile.position}
- **Experience Level:** {profile.experience_years or 'N/A'}

## Objective
{profile.objective}

## Education
"""
    for edu in profile.education:
        page_content += f"""
- **Institution:** {edu.institution}
  - **Degree:** {edu.degree or 'N/A'}, {edu.field_of_study or 'N/A'}
  - **Period:** {edu.start_date or '?'} - {edu.end_date or '?'}
  - **Description:** {edu.description}
"""
    metadata = {
        "apply_doc_type": "candidate_profile",
        "candidate_name": profile.name,
        "objective": profile.objective,
        "position": profile.position,
        "experience_years": profile.experience_years,
    }
    return Document(page_content=page_content, metadata=metadata)


def _convert_career_exp_to_document(
    career: CareerExperience, candidate_name: str
) -> Document:
    """Converts a CareerExperience to a Document."""
    page_content = f"""
# Career Experience: {career.company}

- **Company:** {career.company} ({career.company_description})
- **Position:** {career.position}
- **Job Level:** {career.job_level or 'N/A'}
- **Employment Type:** {career.employee_type}
- **Period:**  {career.start_date or '?'} - {career.end_date or '?'}

## Summary
{career.summary}

## Tech Stack
- {career.tech_stack}

## Achievements (STAR Method)
### Situation
{"- \n".join(career.situation)}
### Task
{"- \n".join(career.task)}
### Action
{"- \n".join(career.action)}
### Result
{"- \n".join(career.result)}
"""
    if career.architecture:
        page_content += f"\n## Architecture\n```mermaid\n{career.architecture}\n```"

    metadata = {
        "apply_doc_type": "career_experience",
        "candidate_name": candidate_name,
        "summary": career.summary,
        "company": career.company,
        "positions": career.position,
        "tech_stack": career.tech_stack,
        "start_date": career.start_date,
        "end_date": career.end_date,
    }
    return Document(page_content=page_content, metadata=metadata)


def _convert_project_exp_to_document(
    project: ProjectExperience, candidate_name: str
) -> Document:
    """Converts a ProjectExperience to a Document."""
    page_content = f"""
# Project Experience: {project.project_name}

- **Project:** {project.project_name}
- **Project Type:** {project.project_type}
- **Team Size:** {project.team_size or 'N/A'}
- **Period:** {project.start_date or '?'} - {project.end_date or '?'}
- **Position:** {project.position}

## Summary
{project.summary}

## Tech Stack
{"- \n".join(project.tech_stack)}

## Contributions (STAR Method)
### Situation
{"- \n".join(project.situation)}
### Task
{"- \n".join(project.task)}
### Action
{"- \n".join(project.action)}
### Result
{"- \n".join(project.result)}
"""
    if project.architecture:
        page_content += f"\n## Architecture\n```mermaid\n{project.architecture}\n```"

    metadata = {
        "apply_doc_type": "project_experience",
        "candidate_name": candidate_name,
        "summary": project.summary,
        "project_name": project.project_name,
        "positions": project.position,
        "tech_stack": project.tech_stack,
        "start_date": project.start_date,
        "end_date": project.end_date,
    }
    return Document(page_content=page_content, metadata=metadata)


def convert_resume_to_documents(
    parsed_result: ResumeParseResult
) -> list[Document]:
    """
    Converts a ResumeParseResult object into a list of Document objects,
    chunking the resume into logical sections.
    """
    documents = []
    candidate_name = parsed_result.candidate_profile.name

    # Convert candidate profile
    documents.append(
        _convert_profile_to_document(parsed_result.candidate_profile)
    )

    # Convert each career experience
    for career in parsed_result.career_experiences:
        documents.append(
            _convert_career_exp_to_document(career, candidate_name)
        )

    # Convert each project experience
    for project in parsed_result.project_experiences:
        documents.append(
            _convert_project_exp_to_document(project, candidate_name)
        )

    return documents 
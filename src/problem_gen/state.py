from dataclasses import dataclass, field
from typing import Optional, Literal, Annotated, operator, TypedDict
import os
from problem_gen.schema import Problem_Content
from constants.metadata import API_VERSION
Problem_Type = Literal["experience", "tech", "cowork"]


@dataclass(kw_only=True)
class Document_with_Id():
    id: str = field(
        default=None, metadata={"description": "The document id."}
    )

    page_content: str = field(
        default=None, metadata={"description": "The document page content."}
    )
    metadata: dict = field(
        default=None, metadata={"description": "The document metadata."}
    )

class Problems(TypedDict):
    problem_type: Problem_Type
    content: Annotated[list[Problem_Content], operator.add]

@dataclass(kw_only=True)
class ProblemGenState:
    user_id: str = field(
        default="b4a80540-b2f8-4b3a-abaf-bed3ecf99f4a",
        metadata={"description": "인증된 사용자 ID"},
    )

    candidate_profile: Document_with_Id = field(
        default_factory=lambda: Document_with_Id(id="d0aff2a7-64ab-42c7-889d-038ebcf3a87d", page_content="", metadata={}), metadata={"description": "The candidate profile."}
    )

    experience: Document_with_Id = field(
        default_factory=lambda: Document_with_Id(id="0cdc466b-ecd6-4010-87d9-70cfc3370016", page_content="", metadata={}), metadata={"description": "The experience."}
    )

    api_version: str = field(
        default=API_VERSION, metadata={"description": "The api version of the schema."}
    )

    problem_type: Problem_Type = field(
        default=None, metadata={"description": "The problem type."}
    )

    problems: Annotated[list[Problems], operator.add] = field(
        default=None, metadata={"description": "The problems."}
    )

    error: Optional[str] = field(
        default=None, metadata={"description": "Any error that occurred during problem generation."}
    )


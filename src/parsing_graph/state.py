from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from parsing_graph.schema.schema import ResumeParseResult
from parsing_graph.schema.is_resume import IsResumeResult
from langchain_core.documents import Document


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


@dataclass(kw_only=True)
class ParsingState:
    """
    State for the parsing graph.
    """

    user_id: str = field(
        default="b4a80540-b2f8-4b3a-abaf-bed3ecf99f4a",
        metadata={"description": "인증된 사용자 ID"},
    )

    resume_file_path: str = field(
        default="b4a80540-b2f8-4b3a-abaf-bed3ecf99f4a/1752622443473_DDD______________________________.pdf",
        metadata={"description": "Supabase Storage 내 파일 경로"},
    )

    resume_content: Optional[str] = field(
        default=None, 
        metadata={"description": "Base64 encoded resume content from Supabase Storage"}
    )

    is_resume_result: Optional[IsResumeResult] = field(
        default=None,
        metadata={"description": "The result of checking if the document is a resume."},
    )

    parsed_result: Optional[ResumeParseResult] = field(
        default=None, metadata={"description": "The parsed result from the resume."}
    )

    documents: list[Document] = field(
        default_factory=list,
        metadata={"description": "The parsed resume converted to documents."},
    )
    
    error: Optional[str] = field(
        default=None, metadata={"description": "Any error that occurred during parsing."}
    )
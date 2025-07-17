from dataclasses import dataclass, field
from typing import Optional
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
        default="https://tujkavzwiiljzkxotfzi.supabase.co/storage/v1/object/sign/resumes/b4a80540-b2f8-4b3a-abaf-bed3ecf99f4a/1752740568963_DDD______________________________.pdf?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV9mMmQ0N2Q4Ny01NTNlLTRlYzYtOTcyOS1mODRhYWQ1NmE0ZDUiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJyZXN1bWVzL2I0YTgwNTQwLWIyZjgtNGIzYS1hYmFmLWJlZDNlY2Y5OWY0YS8xNzUyNzQwNTY4OTYzX0RERF9fX19fX19fX19fX19fX19fX19fX19fX19fX19fXy5wZGYiLCJpYXQiOjE3NTI3NDA2MjQsImV4cCI6MTc4NDI3NjYyNH0.2hd1njHf5--9IyOofkzsFjgnzvbiPJT-N1FXpnvK0NQ",
        metadata={"description": "Supabase Storage 내 파일 경로"},
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

    parse_retry_count: int = field(
        default=0, metadata={"description": "The number of parsing attempts."}
    )
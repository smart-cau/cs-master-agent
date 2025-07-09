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


# Define the project root and the default file path relative to this file's location.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILE_PATH = _PROJECT_ROOT / "static" / "DDD_정현우_백엔드_포트폴리오.pdf"
NON_RESUME_FILE_PATH = _PROJECT_ROOT / "static" / "non-resume.pdf"


@dataclass(kw_only=True)
class ParsingState:
    """
    State for the parsing graph.
    """

    file_path: str = field(
        default=str(DEFAULT_FILE_PATH),
        metadata={"description": "The path to the file to be parsed."},
    )

    user_id: str = field(
        default="정현우", metadata={"description": "The user id of the user."}
    )

    document_content: Optional[str] = field(
        default=None, metadata={"description": "The text content of the document."}
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
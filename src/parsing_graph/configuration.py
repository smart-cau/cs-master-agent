"""Define the configurable parameters for the agent."""

# Python 3.7+ 에서 타입 힌팅을 위한 forward reference 기능을 활성화합니다.
# 이 import를 통해 클래스 정의 내에서 자기 자신의 타입을 참조할 수 있습니다.
# 예: @classmethod 메서드가 자신의 클래스 타입(ConfigSchema)을 반환할 때 사용됩니다.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.config import get_config

from parsing_graph.prompts import PARSING_SYSTEM_PROMPT, IS_RESUME_SYSTEM_PROMPT


@dataclass(kw_only=True)
class ConfigSchema:
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """
    career_relevant_document_parse_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default='gemini-2.5-pro',
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )
    is_resume_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default='gemini-2.0-flash',
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    temperature: float = field(
        default=0.1,
        metadata={
            "description": "The temperature of the language model to use for the agent's main interactions. "
            "0.0 is the most deterministic output."
        },
    )

    system_prompt: str = field(
        default=PARSING_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    is_resume_system_prompt: str = field(
        default=IS_RESUME_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    user_service_id: str = field(
        default="정현우",
        metadata={
            "description": "The user service id of the user."
        },
    )

    @classmethod
    def from_runnable_config(cls: Type[T], config: Optional[RunnableConfig] = None) -> T:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
    

T = TypeVar("T", bound=ConfigSchema)
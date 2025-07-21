from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from langgraph.config import get_config

from problem_gen.prompts import EXPERIENCE_PROBLEM_GEN_SYSTEM_PROMPT, TECH_PROBLEM_GEN_SYSTEM_PROMPT, COWORK_PROBLEM_GEN_SYSTEM_PROMPT, BASE_SYSTEM_PROMPT
    

@dataclass(kw_only=True)
class ConfigSchema:

    experience_problem_gen_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default='google_genai:gemini-2.5-flash',
        metadata={
            "description": "사용자의 업무/프로젝트 경험과 관련한 문제를 생성할 때 사용되는 모델"
            "Should be in the form = provider:model-name."
        },
    )

    tech_problem_gen_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default='anthropic:claude-sonnet-4-20250514',
        default='google_genai:gemini-2.5-flash',
        metadata={
            "description": "사용자의 사용 기술과 관련한 문제를 생성할 때 사용되는 모델"
            "Should be in the form = provider:model-name."
        },
    )

    cowork_problem_gen_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default='google_genai:gemini-2.5-flash',
        metadata={
            "description": "사용자의 협업 경험과 관련한 문제를 생성할 때 사용되는 모델"
            "Should be in the form = provider:model-name."
        },
    )

    base_system_prompt: str = field(
        default=BASE_SYSTEM_PROMPT,
        metadata={
            "description": "문제를 생성할 때 사용되는 기본 시스템 프롬프트"
        },
    )

    experience_problem_gen_system_prompt: str = field(
        default=EXPERIENCE_PROBLEM_GEN_SYSTEM_PROMPT,
        metadata={
            "description": "사용자의 업무/프로젝트 경험과 관련한 문제를 생성할 때 사용되는 시스템 프롬프트"
        },
    )

    tech_problem_gen_system_prompt: str = field(
        default=TECH_PROBLEM_GEN_SYSTEM_PROMPT,
        metadata={
            "description": "사용자의 사용 기술과 관련한 문제를 생성할 때 사용되는 시스템 프롬프트"
        },
    )

    cowork_problem_gen_system_prompt: str = field(
        default=COWORK_PROBLEM_GEN_SYSTEM_PROMPT,
        metadata={
            "description": "사용자의 협업 경험과 관련한 문제를 생성할 때 사용되는 시스템 프롬프트"
        },
    )

    problem_gen_temperature: float = field(
        default=0.1,
        metadata={
            "description": "temperature of the model that generate problem"
        },
    )

    timeout: int = field(
        default=100,
        metadata={
            "description": "timeout of the model that generate problem"
        },
    )

    max_retries: int = field(
        default=2,
        metadata={
            "description": "max retries of the model that generate problem"
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
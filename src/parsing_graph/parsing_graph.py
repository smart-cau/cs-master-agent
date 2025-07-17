import logging
import langsmith
from uuid import uuid4
from typing import Dict, Any

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from parsing_graph.configuration import ConfigSchema
from parsing_graph.schema.schema import ResumeParseResult
from parsing_graph.schema.is_resume import IsResumeResult
from parsing_graph.state import ParsingState
from parsing_graph.converter import convert_resume_to_documents
from parsing_graph.vector_store import vector_store, delete_docs_by

langsmith_logger = logging.getLogger("langsmith")
langsmith_logger.setLevel(logging.DEBUG)

MAX_PARSE_RETRIES = 2


def is_resume_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines if the document is a resume.
    """

    try:
        configurable = ConfigSchema.from_runnable_config(config)
        model_name = configurable.is_resume_model
        temperature = configurable.temperature
        system_prompt = configurable.is_resume_system_prompt
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, timeout=100)
        structured_llm = llm.with_structured_output(IsResumeResult)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Please check if the attached document is a job application related document."},
                    {"type": "image_url", "image_url": {"url": state.resume_file_path}},
                ]
            ),
        ]

        is_resume_result = structured_llm.invoke(messages, config)
        return {
            "is_resume_result": is_resume_result,
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return {
                "is_resume_result": None,
                "error": f"AI 모델 할당량 초과 또는 요청 제한: {error_msg}",
            }
        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "is_resume_result": None,
                "error": f"AI 모델 인증 오류: {error_msg}",
            }
        else:
            return {
                "is_resume_result": None,
                "error": f"이력서 여부 판단 중 오류가 발생했습니다: {error_msg}",
            }


def parse_resume_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Parses the document using a multimodal LLM.
    """
    try:
        configurable = ConfigSchema.from_runnable_config(config)
        model_name = configurable.career_relevant_document_parse_model
        temperature = configurable.temperature
        system_prompt = configurable.system_prompt
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, thinking_budget=8192, max_output_tokens=8192, timeout=100)
        structured_llm = llm.with_structured_output(ResumeParseResult)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Please parse the attached resume PDF and extract the information based on the `ResumeParseResult` schema.",
                    },
                    {"type": "image_url", "image_url": {"url": state.resume_file_path}},
                ]
            ),
        ]

        parsed_result = structured_llm.invoke(messages, config)
        return {
            "parsed_result": parsed_result,
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        langsmith_logger.error(f"Error parsing resume: {error_msg}")
        if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return {
                "parsed_result": None,
                "error": f"AI 모델 할당량 초과 또는 요청 제한: {error_msg}",
            }
        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "parsed_result": None,
                "error": f"AI 모델 인증 오류: {error_msg}",
            }
        elif "timeout" in error_msg.lower():
            return {
                "parsed_result": None,
                "error": f"AI 모델 요청 시간 초과: {error_msg}",
            }
        else:
            return {
        
                "parsed_result": None,
                "error": f"문서 파싱 중 오류가 발생했습니다: {error_msg}",
            }


def should_parse_resume(state: ParsingState, config: RunnableConfig) -> str:
    """
    Determines whether to parse the document or end the process.
    """
    if state.error:
        return "clean_up"

    if state.is_resume_result and state.is_resume_result.is_resume:
        return "parse_resume"
    else:
        print(f"Document is not a resume. Reason: {state.is_resume_result.reason if state.is_resume_result else 'Unknown'}")
        return "clean_up"


def should_convert_to_document(state: ParsingState) -> str:
    """
    파싱된 결과를 문서로 변환할지, 파싱을 재시도할지, 아니면 종료할지 결정합니다.
    """
    if state.error:
        # parse_resume_node에서 발생한 오류 처리
        langsmith_logger.warning(f"Parsing failed with error: {state.error}. Checking for retries.")

    if not state.parsed_result:
        if state.parse_retry_count < MAX_PARSE_RETRIES:
            langsmith_logger.info(f"Retrying parsing. Attempt {state.parse_retry_count + 1}/{MAX_PARSE_RETRIES + 1}")
            return "retry_parse"
        else:
            langsmith_logger.error(f"Max parsing retries reached ({MAX_PARSE_RETRIES}). Cleaning up.")
            return "clean_up"

    return "convert_to_document"


def handle_parse_failure_node(state: ParsingState) -> Dict[str, Any]:
    """
    파싱 재시도 횟수를 증가시키고 다음 재시도를 위해 상태를 정리합니다.
    """
    langsmith_logger.error(f"Error parsing resume: {state.error}. Checking for retries.")
    return {
        "parse_retry_count": state.parse_retry_count + 1,
        "error": None,  # 이전 오류를 지우고 재시도
    }


def parsed_resume_to_document_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Converts the parsed result to a langchain document.
    """
    try:
        parsed_result = state.parsed_result
        if not parsed_result:
            # This path should not be taken due to the conditional edge, but it remains as a safeguard.
            return {"documents": None, "error": "Defensive check failed: parsed_result is empty in parsed_resume_to_document_node."}

        documents = convert_resume_to_documents(parsed_result=parsed_result)

        for doc in documents:
            doc.metadata["file_path"] = state.resume_file_path
            doc.metadata["user_id"] = state.user_id

        return {
            "documents": documents,
            "error": None,
        }
    except Exception as e:
        langsmith_logger.error(f"Error converting parsed result to documents: {str(e)}")
        return {
            "documents": None,
            "error": f"파싱 결과를 문서로 변환하는 중 오류가 발생했습니다: {str(e)}",
        }


def add_documents_to_qdrant_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Adds the documents to the Qdrant vector store.
    """
    if not state.documents:
        return {
            "error": "벡터 스토어에 추가할 문서가 없습니다.",
        }

    try:
        langsmith_logger.debug(f"DEBUG: try to delete docs by user_id: {state.user_id}")
        # 기존 사용자 문서 삭제
        delete_docs_by(key="metadata.user_id", value=state.user_id)
        langsmith_logger.debug(f"DEBUG: delete docs by user_id: {state.user_id}")
        
        # 새 문서 추가
        uuids = [str(uuid4()) for _ in range(len(state.documents))]
        langsmith_logger.debug(f"DEBUG: try to add docs to qdrant: {state.documents[0].metadata}")
        vector_store.add_documents(documents=state.documents, ids=uuids)
        langsmith_logger.debug(f"DEBUG: add docs to qdrant: {state.documents[0].metadata}")
        return {
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return {
                "error": f"벡터 데이터베이스 연결 오류: {error_msg}",
            }
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return {
                "error": f"벡터 데이터베이스 인증 오류: {error_msg}",
            }
        elif "embedding" in error_msg.lower():
            return {
                "error": f"문서 임베딩 생성 오류: {error_msg}",
            }
        else:
            return {
                "error": f"벡터 스토어에 문서 추가 중 오류가 발생했습니다: {error_msg}",
            }

def clean_up_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    return {
        "documents": None,
    }

graph_builder = StateGraph(ParsingState, config_schema=ConfigSchema)

"""NODES"""
graph_builder.add_node("is_resume", is_resume_node)
graph_builder.add_node("parse_resume", parse_resume_node)
graph_builder.add_node("parsed_resume_to_document", parsed_resume_to_document_node)
graph_builder.add_node("add_documents_to_qdrant", add_documents_to_qdrant_node)
graph_builder.add_node("clean_up", clean_up_node)
graph_builder.add_node("handle_parse_failure", handle_parse_failure_node)

"""EDGES"""
graph_builder.add_edge(START, "is_resume")

graph_builder.add_conditional_edges(
    "is_resume",
    should_parse_resume,
    {
        "parse_resume": "parse_resume",
        "clean_up": "clean_up",
    },
)
graph_builder.add_conditional_edges(
    "parse_resume",
    should_convert_to_document,
    {
        "convert_to_document": "parsed_resume_to_document",
        "retry_parse": "handle_parse_failure",
        "clean_up": "clean_up"
    }
)
graph_builder.add_edge("handle_parse_failure", "parse_resume")
graph_builder.add_edge("parsed_resume_to_document", "add_documents_to_qdrant")
graph_builder.add_edge("add_documents_to_qdrant", "clean_up")
graph_builder.add_edge("clean_up", END)

"""COMPILE"""
parsing_graph = graph_builder.compile()
parsing_graph.name = "ParsingGraph"

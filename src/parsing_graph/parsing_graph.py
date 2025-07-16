import base64
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
from parsing_graph.vector_store import get_vector_store, delete_docs_by
from parsing_graph.supabase_utils import (
    validate_user_access_to_file, 
    download_resume_as_base64,
    SupabaseError,
    FileNotFoundError,
    FileDownloadError,
    FileAccessError
)


def load_resume_node(state: ParsingState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    Loads the resume content from Supabase Storage using the file path.
    """
    try:
        # 1. 사용자 권한 검증
        if not validate_user_access_to_file(state.user_id, state.resume_file_path):
            return {
                "error": "파일에 접근 권한이 없거나 파일이 존재하지 않습니다.",
            }
        
        # 2. Storage에서 파일 다운로드 및 Base64 인코딩
        encoded_content = download_resume_as_base64(state.resume_file_path)
        
        if not encoded_content:
            return {
                "error": "파일 다운로드에 실패했습니다.",
            }
        
        return {
            "resume_content": encoded_content,
            "error": None,
        }
    except FileNotFoundError as e:
        return {
            "error": f"파일을 찾을 수 없습니다: {str(e)}",
        }
    except FileDownloadError as e:
        return {
            "error": f"파일 다운로드 실패: {str(e)}",
        }
    except FileAccessError as e:
        return {
            "error": f"파일 접근 권한 오류: {str(e)}",
        }
    except SupabaseError as e:
        return {
            "error": f"Supabase 연결 오류: {str(e)}",
        }
    except Exception as e:
        return {
            "error": f"예상치 못한 오류가 발생했습니다: {str(e)}",
        }


def is_resume_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Determines if the document is a resume.
    """

    if not state.resume_content:
        return {
            "is_resume_result": IsResumeResult(is_resume=False, reason="No document content to parse."),
            "error": "문서 내용이 없어 이력서 여부를 판단할 수 없습니다.",
        }

    try:
        configurable = ConfigSchema.from_runnable_config(config)
        model_name = configurable.is_resume_model
        temperature = configurable.temperature
        system_prompt = configurable.is_resume_system_prompt
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        structured_llm = llm.with_structured_output(IsResumeResult)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Please check if the attached document is a job application related document."},
                    {"type": "media", "mime_type": "application/pdf", "data": state.resume_content},
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
    if not state.resume_content:
        return {
            "parsed_result": None,
            "error": "문서 내용이 없어 파싱할 수 없습니다.",
        }

    try:
        configurable = ConfigSchema.from_runnable_config(config)
        model_name = configurable.career_relevant_document_parse_model
        temperature = configurable.temperature
        system_prompt = configurable.system_prompt
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, thinking_budget=8192, max_output_tokens=8192)
        structured_llm = llm.with_structured_output(ResumeParseResult)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Please parse the attached resume PDF and extract the information based on the `ResumeParseResult` schema.",
                    },
                    {"type": "media", "mime_type": "application/pdf", "data": state.resume_content},
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
        return END

    if state.is_resume_result and state.is_resume_result.is_resume:
        return "parse_resume"
    else:
        print(f"Document is not a resume. Reason: {state.is_resume_result.reason if state.is_resume_result else 'Unknown'}")
        return END


def parsed_resume_to_document_node(state: ParsingState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Converts the parsed result to a langchain document.
    """
    if not state.parsed_result:
        return {
            "documents": None,
            "error": "파싱된 결과가 없어 문서를 생성할 수 없습니다.",
        }

    try:
        parsed_result = state.parsed_result

        documents = convert_resume_to_documents(parsed_result=parsed_result)

        for doc in documents:
            doc.metadata["source"] = state.resume_file_path
            doc.metadata["user_id"] = state.user_id

        return {
            "documents": documents,
            "error": None,
        }
    except Exception as e:
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
        # 기존 사용자 문서 삭제
        delete_docs_by(key="metadata.user_id", value=state.user_id)
        
        # 새 문서 추가
        uuids = [str(uuid4()) for _ in range(len(state.documents))]
        get_vector_store().add_documents(documents=state.documents, ids=uuids)
        
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


graph_builder = StateGraph(ParsingState, config_schema=ConfigSchema)

graph_builder.add_node("load_resume", load_resume_node)
graph_builder.add_node("is_resume", is_resume_node)
graph_builder.add_node("parse_resume", parse_resume_node)
graph_builder.add_node("parsed_resume_to_document", parsed_resume_to_document_node)
graph_builder.add_node("add_documents_to_qdrant", add_documents_to_qdrant_node)

graph_builder.add_edge(START, "load_resume")
graph_builder.add_edge("load_resume", "is_resume")
graph_builder.add_conditional_edges(
    "is_resume",
    should_parse_resume,
    {
        "parse_resume": "parse_resume",
        END: END,
    },
)
graph_builder.add_edge("parse_resume", "parsed_resume_to_document")
graph_builder.add_edge("parsed_resume_to_document", "add_documents_to_qdrant")
graph_builder.add_edge("add_documents_to_qdrant", END)

parsing_graph = graph_builder.compile()
parsing_graph.name = "ParsingGraph"

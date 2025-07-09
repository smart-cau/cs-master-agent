import base64
from uuid import uuid4

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from parsing_graph.configuration import ConfigSchema
from parsing_graph.schema.schema import ResumeParseResult
from parsing_graph.schema.is_resume import IsResumeResult
from parsing_graph.state import ParsingState
from parsing_graph.prompts import PARSING_SYSTEM_PROMPT, IS_RESUME_SYSTEM_PROMPT
from parsing_graph.converter import convert_resume_to_documents
from parsing_graph.vector_store import vector_store, apply_docs_collection_name, delete_docs_by



def load_resume_node(state: ParsingState) -> ParsingState:
    """
    Loads the content of the PDF file specified in the state.
    """
    try:
        with open(state.file_path, "rb") as f:
            document_content = f.read()
        return ParsingState(
            file_path=state.file_path, document_content=document_content, error=None
        )
    except Exception as e:
        return ParsingState(
            file_path=state.file_path,
            document_content=None,
            error=f"Failed to load file: {e}",
        )


def is_resume_node(state: ParsingState, config: RunnableConfig) -> ParsingState:
    """
    Determines if the document is a resume.
    """
    if state.error:
        return state

    document_content = state.document_content
    if not document_content:
        return ParsingState(
            file_path=state.file_path,
            is_resume_result=IsResumeResult(is_resume=False, reason="No document content to parse."),
            error="No document content to parse.",
        )

    encoded_content = base64.b64encode(document_content).decode("utf-8")

    configurable = config.get("configurable", {})
    model_name = configurable.get("is_resume_model", "gemini-2.0-flash")
    temperature = configurable.get("temperature", 0.1)
    system_prompt = configurable.get("is_resume_system_prompt", IS_RESUME_SYSTEM_PROMPT)
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(IsResumeResult)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": "Please check if the attached document is a job application related document."},
                {"type": "media", "mime_type": "application/pdf", "data": encoded_content},
            ]
        ),
    ]

    try:
        is_resume_result = structured_llm.invoke(messages, config)
        return ParsingState(
            file_path=state.file_path,
            document_content=state.document_content,
            is_resume_result=is_resume_result,
            error=None,
        )
    except Exception as e:
        return ParsingState(
            file_path=state.file_path,
            document_content=state.document_content,
            is_resume_result=None,
            error=f"Failed to check if document is a resume: {e}",
        )


def parse_resume_node(state: ParsingState, config: RunnableConfig) -> ParsingState:
    """
    Parses the document using a multimodal LLM.
    """
    if state.error:
        return state

    document_content = state.document_content
    if not document_content:
        return ParsingState(
            file_path=state.file_path,
            parsed_result=None,
            error="No document content to parse.",
        )

    encoded_content = base64.b64encode(document_content).decode("utf-8")

    # The config is passed in from the .invoke() call
    configurable = config.get("configurable", {})
    model_name = configurable.get("career_relevant_document_parse_model", "gemini-2.5-pro")
    temperature = configurable.get("temperature", 0.1)
    system_prompt = configurable.get("system_prompt", PARSING_SYSTEM_PROMPT)    
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
                {"type": "media", "mime_type": "application/pdf", "data": encoded_content},
            ]
        ),
    ]

    try:
        # Pass the config down to the LLM
        parsed_result = structured_llm.invoke(messages, config)
        state.parsed_result = parsed_result
        return state
    except Exception as e:
        state.error = f"Failed to parse document: {e}"
        return state


def should_parse_resume(state: ParsingState) -> str:
    """
    Determines whether to parse the document or end the process.
    """
    if state.error:
        return END
    
    if state.is_resume_result and state.is_resume_result.is_resume:
        return "parse_resume"
    else:
        # You can also log the reason here if you want
        print(f"Document is not a resume. Reason: {state.is_resume_result.reason}")
        return END

def parsed_resume_to_document_node(state: ParsingState) -> ParsingState:
    """
    Converts the parsed result to a langchain document.
    """
    if state.error or not state.parsed_result:
        return state

    parsed_result = state.parsed_result
    
    # Convert parsed_result to a list of Document objects
    documents = convert_resume_to_documents(
        parsed_result=parsed_result
    )

    # add 'source' and 'user_id'
    for doc in documents:
        doc.metadata["source"] = state.file_path
        doc.metadata["user_id"] = state.user_id
    
    state.documents = documents
    return state


def add_documents_to_qdrant_node(state: ParsingState) -> ParsingState:
    """
    Adds the documents to the Qdrant vector store.
    """
    if state.error or not state.documents:
        return state

    try:
        # check whether the docs with metadata 'user_id' already exist in the vector store
        delete_docs_by(key="metadata.user_id", value=state.user_id)

        uuids = [str(uuid4()) for _ in range(len(state.documents))]
        print(f"DEBUG: length of documents: {len(state.documents)}")
        print(f"DEBUG: length of uuids: {len(uuids)}")
        
        vector_store.add_documents(documents=state.documents, ids=uuids)

        print(f"Successfully added {len(state.documents)} documents to Qdrant collection '{apply_docs_collection_name}'.")
    except Exception as e:
        state.error = f"Failed to add documents to Qdrant: {e}"
    
    return state


graph_builder = StateGraph(ParsingState, config_schema=ConfigSchema)

graph_builder.add_node("load_resume", load_resume_node)
graph_builder.add_node("is_resume", is_resume_node)
graph_builder.add_node("parse_resume", parse_resume_node)
graph_builder.add_node("parsed_resume_to_document", parsed_resume_to_document_node)
graph_builder.add_node("add_documents_to_qdrant", add_documents_to_qdrant_node)

graph_builder.set_entry_point("load_resume")
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

# The graph is configurable with the ConfigSchema
parsing_graph = graph_builder.compile() 
parsing_graph.name = "ParsingGraph"
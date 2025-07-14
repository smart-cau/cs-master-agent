from langchain_core.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool

from resume_chat_graph.retriever import get_retriever_for_user
from resume_chat_graph.configuration import ConfigSchema


@tool(name_or_callable="retreive_user_apply_docs_tool", description="Searches and returns excerpts from the user's resume and career documents. Use it to answer questions about the user's experience, projects, and skills.")
def retreive_user_apply_docs_tool(query: str, user_service_id: str)-> str:
    retriever = get_retriever_for_user(user_service_id)
    docs = retriever.invoke(query)

    return "\n".join([doc.page_content for doc in docs])
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig

from resume_chat_graph.state import State, InputState
from resume_chat_graph.configuration import ConfigSchema
from resume_chat_graph.retriever import get_retriever_for_user
from resume_chat_graph.utils import get_message_text
from resume_chat_graph.schema import GeneratedQueries


def transform_query_node(state: State, config: RunnableConfig) -> dict:
    """Transforms the user's query into a set of optimized search queries."""
    
    configuration = ConfigSchema.from_runnable_config(config)
    print(f"DEBUG: configuration: {configuration}")

    query_gen_llm = ChatGoogleGenerativeAI(
        model=configuration.query_model, temperature=0
    ).with_structured_output(GeneratedQueries)

    user_question = get_message_text(state.messages[-1])
    system_prompt = configuration.query_system_prompt

    prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Generate search queries for the following user question: {user_question}"
        ),
    ]
    generated_queries = query_gen_llm.invoke(prompt, config)
    print(f"DEBUG: generated_queries: {generated_queries}")
    return {"queries": generated_queries.queries}


def retrieve_docs_node(state: State, config: RunnableConfig) -> dict:
    """Retrieves documents based on the transformed queries."""
    retriever = get_retriever_for_user(state.user_id)
    retrieved_docs = retriever.batch(state.queries, config)
    # Flatten the list of lists and remove duplicates
    unique_docs = {doc.page_content: doc for docs in retrieved_docs for doc in docs}.values()
    return {"retrieved_docs": list(unique_docs)}


def generate_response_node(state: State, config: RunnableConfig) -> dict:
    """Generates a response based on the retrieved documents and user query.""" 
    configuration = ConfigSchema.from_runnable_config(config)
    response_llm = ChatGoogleGenerativeAI(model=configuration.response_model, temperature=0.1)

    system_prompt = configuration.response_system_prompt
    user_question = get_message_text(state.messages[-1])
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in state.retrieved_docs]
    )
    
    prompt = f"{system_prompt}\n\nHere is the retrieved context:\n\n{context}\n\nUser Question: {user_question}"
    
    response = response_llm.invoke([HumanMessage(content=prompt)], config)
    return {"messages": [response]}


def cannot_answer_node(state: State, config: RunnableConfig) -> dict:
    """Generates a message indicating that no relevant information was found."""
    response = HumanMessage(
        content="I'm sorry, but I couldn't find any relevant information in your documents to answer your question. Please try asking something else."
    )
    return {"messages": [response]}


def should_generate_edge(state: State) -> str:
    """Determines whether to generate a response or indicate that no answer can be found."""
    if state.retrieved_docs:
        return "generate_response"
    else:
        return "cannot_answer"


builder = StateGraph(State, input=InputState, config_schema=ConfigSchema)

builder.add_node("transform_query", transform_query_node)
builder.add_node("retrieve_docs", retrieve_docs_node)
builder.add_node("generate_response", generate_response_node)
builder.add_node("cannot_answer", cannot_answer_node)

builder.add_edge(START, "transform_query")
builder.add_edge("transform_query", "retrieve_docs")
builder.add_conditional_edges(
    "retrieve_docs",
    should_generate_edge,
    {"generate_response": "generate_response", "cannot_answer": "cannot_answer"},
)
builder.add_edge("generate_response", END)
builder.add_edge("cannot_answer", END)


memory = SqliteSaver.from_conn_string(":memory:")
resume_chat_graph = builder.compile(checkpointer=memory)
resume_chat_graph.name = "resume_chat_graph"
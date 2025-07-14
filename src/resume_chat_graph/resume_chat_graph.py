from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import tools_condition, ToolNode

from resume_chat_graph.state import State, InputState
from resume_chat_graph.configuration import ConfigSchema
from resume_chat_graph.utils import get_message_text
from resume_chat_graph.schema import GeneratedQueries
from datetime import datetime
from resume_chat_graph.tools import retreive_user_apply_docs_tool

tools=[retreive_user_apply_docs_tool]   

async def chat_node(state: State, config: RunnableConfig) -> dict:
    """Chat with the user."""
    configuration = ConfigSchema.from_runnable_config(config)
    response_llm = ChatGoogleGenerativeAI(model=configuration.response_model, temperature=0.1).bind_tools(tools)
    system_prompt = configuration.response_system_prompt
    
    system_prompt = system_prompt.format(user_service_id=configuration.user_service_id, system_time=datetime.now().isoformat())

    messages = [
        SystemMessage(content=system_prompt),
        *state.messages,
    ]
    
    response = await response_llm.ainvoke(messages, config)
    messages.append(response)
    
    return {"messages": messages}


    

def transform_query_node(state: State, config: RunnableConfig) -> dict:
    """Transforms the user's query into a set of optimized search queries."""
    
    configuration = ConfigSchema.from_runnable_config(config)

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


builder = StateGraph(State, input=InputState, config_schema=ConfigSchema)

"""Nodes"""
builder.add_node("chat", chat_node)
builder.add_node("tools", ToolNode(tools=tools))
builder.add_node("transform_query", transform_query_node)
# builder.add_node("retrieve_docs", retrieve_docs_node)
builder.add_node("generate_response", generate_response_node)
# builder.add_node("cannot_answer", cannot_answer_node)

""" Edges """
builder.add_edge(START, "chat")
builder.add_conditional_edges('chat', tools_condition)
builder.add_edge('tools', 'chat')
# builder.add_edge("chat", "transform_query")
# builder.add_edge("transform_query", "retrieve_docs")
# builder.add_conditional_edges(
#     "retrieve_docs",
#     should_generate_edge,
#     {"generate_response": "generate_response", "cannot_answer": "cannot_answer"},
# )
# builder.add_edge("generate_response", END)
# builder.add_edge("cannot_answer", END)
builder.add_edge("chat", END)


# memory = SqliteSaver.from_conn_string(":memory:")
# resume_chat_graph = builder.compile(checkpointer=memory)
resume_chat_graph = builder.compile()
resume_chat_graph.name = "resume_chat_graph"

import langsmith
import logging
from typing import Dict, Any
from uuid import uuid4
from langgraph.graph import END, StateGraph, START
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from problem_gen.schema import Problem_Contents
from problem_gen.state import ProblemGenState, Problem_Type, Problems
from problem_gen.config import ConfigSchema
from constants.vector_store import personalized_problems_vector_store, apply_docs_vector_store, delete_docs_by, personalized_problems_collection_name

# Loggers are hierarchical, so setting the log level on "langsmith" will
# set it on all modules inside the "langsmith" package
langsmith_logger = logging.getLogger("langsmith")
langsmith_logger.setLevel(level=logging.DEBUG)

def load_documents(state: ProblemGenState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Load documents from vector store.
    """
    """
        TODO: 문서 load logic 구현
        - vector_store.get_by_id() 사용
        - state.candidate_profile.id, state.experience.id를 사용해서 조회
        - 조회된 문서를 state.candidate_profile, state.experience에 저장(id로 매핑해서)
            - 조회된 문서는 id 순서대로 오는 것이 아니기에, id를 매핑해서 할당해야 함   
    """
    docs: list[Document] = apply_docs_vector_store.get_by_ids(
        [state.candidate_profile.id, state.experience.id]
    )

    # 문서를 유형별로 분류
    candidate_profile_doc = None
    experience_doc = None
    
    for doc in docs:
        doc_type = doc.metadata.get("apply_doc_type", "")
        if doc_type == "candidate_profile":
            candidate_profile_doc = doc
        elif "experience" in doc_type:
            experience_doc = doc

    if not candidate_profile_doc or not experience_doc:
        raise ValueError("Candidate profile or experience not found")
    
    profile = state.candidate_profile
    profile.page_content = candidate_profile_doc.page_content
    profile.metadata = candidate_profile_doc.metadata

    experience = state.experience
    experience.page_content = experience_doc.page_content
    experience.metadata = experience_doc.metadata

    return {
        "candidate_profile": profile,
        "experience": experience
    }

def assign_workers(state: ProblemGenState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Assign workers.
    """

    # list of problem_gen_configs
    problem_gen_configs: list[Problem_Type] = ["experience", "tech", "cowork"]

    return [Send("problem_gen", {"problem_type": problem_type, "candidate_profile": state.candidate_profile, "experience": state.experience}) for problem_type in problem_gen_configs]

# TODO: 특정 LLM 모델이 사용량 초과로 인해 문제 생성 실패 시, 다른 모델로 대체하는 로직 추가
def problem_gen(state: ProblemGenState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Generate problem.
    """
    configuration = ConfigSchema.from_runnable_config(config)

    problem_type = state.get('problem_type')
    candidate_profile = state.get('candidate_profile')
    experience = state.get('experience')

    base_system_prompt = configuration.base_system_prompt.format(
        candidate_name=candidate_profile.metadata.get("candidate_name", ""),
        position=candidate_profile.metadata.get("position", ""),
        objective=candidate_profile.metadata.get("objective", ""),
        experience=experience.page_content
    )

    problem_type_system_prompt = ""
    
    match problem_type:
        case "experience":
            model = configuration.experience_problem_gen_model
            problem_type_system_prompt = configuration.experience_problem_gen_system_prompt
        case "tech":
            model = configuration.tech_problem_gen_model
            problem_type_system_prompt = configuration.tech_problem_gen_system_prompt
        case "cowork":
            model = configuration.cowork_problem_gen_model
            problem_type_system_prompt = configuration.cowork_problem_gen_system_prompt
        case _:
            raise ValueError(f"Invalid problem type: {problem_type}")
        
    system_prompt = base_system_prompt + problem_type_system_prompt
        
    problem_gen_model = init_chat_model(model=model, temperature=configuration.problem_gen_temperature, timeout=configuration.timeout, max_retries=configuration.max_retries)
    structured_problem_gen_model = problem_gen_model.with_structured_output(Problem_Contents)

    messages= [
        SystemMessage(content=system_prompt),
        HumanMessage(content=[
            {"type": "text", "text": "Please generate problems based on the following resume. Response in Korean."},
        ])
    ]
    
    problem_contents = structured_problem_gen_model.invoke(messages)
    

    return {
        "problems": [Problems(problem_type=problem_type, content=problem_contents.contents)]
    }


def gather_all_problems(state: ProblemGenState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Gather all problems.
    """
    personalized_problems_vector_store.delete(ids=[state.experience.id])

    problem_docs = []
    for problems_with_type in state.problems:
        for problem_content in problems_with_type['content']:
            problem_doc = Document(
                page_content=f"question:{problem_content.question}\nexplanation:{problem_content.explanation}",
                metadata={
                    "problem_type": problems_with_type['problem_type'],
                    "user_id": state.user_id,
                    "api_version": state.api_version,
                    "experience_id": state.experience.id,
                }
            )
            problem_docs.append(problem_doc)

    uuids = [str(uuid4()) for _ in range(len(problem_docs))]
    personalized_problems_vector_store.add_documents(documents=problem_docs, ids=uuids)

    return {
        "problems": state.problems
    }

"""GRAPH BUILDER"""
graph_builder = StateGraph(ProblemGenState, config_schema=ConfigSchema)

"""NODES"""
graph_builder.add_node("load_documents", load_documents)
graph_builder.add_node("problem_gen", problem_gen)
graph_builder.add_node("gather_all_problems", gather_all_problems)

"""EDGES"""
graph_builder.add_edge(START, "load_documents")
graph_builder.add_conditional_edges("load_documents", assign_workers, ["problem_gen"])
graph_builder.add_edge("problem_gen", "gather_all_problems")
graph_builder.add_edge("gather_all_problems", END)

"""COMPILE"""
problem_gen_graph = graph_builder.compile()
problem_gen_graph.name = "ProblemGenGraph"

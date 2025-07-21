
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
    

    return {
        "candidate_profile": candidate_profile_doc,
        "experience": experience_doc
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
    langsmith_logger.debug(f"DEBUG: GETHER_NODE -> state.problems: {state.problems}")
    """
        state.problems = [
        {'problem_type': 'experience', 
        'content': [Problem_Content(question='"주변시위 Now" 프로젝트를 시작하게 된 배경과 궁극적으로 달성하고자 했던 목표는 무엇이었는지 설명해 주시겠어요?', explanation='지원자님의 프로젝트 전반에 대한 이해도와 문제 해결 역량을 파악하기 위함입니다.'), Problem_Content(question='"주변시위 Now" 프로젝트에서 팀 리더이자 백엔드 개발자로서 구체적으로 어떤 역할을 수행하셨고, 팀원들과의 협업은 어떻게 이끌어 나가셨나요?', explanation='팀 리더로서의 역할과 백엔드 개발자로서의 기여, 그리고 팀워크 및 리더십 역량을 확인하기 위함입니다.'), Problem_Content(question='K6 성능 테스트 후 RPS 150배 증가, 응답시간 72.4% 개선이라는 놀라운 성과를 달성하셨는데, 이 중 가장 큰 효과를 보았던 개선점(예: Redis Cache, Connection Pool 튜닝)은 무엇이었고 그 이유는 무엇인가요?', explanation='성능 최적화 과정에서의 문제 해결 능력과 기술적 깊이를 확인하고, 가장 큰 임팩트를 낸 부분에 대한 구체적인 설명을 듣기 위함입니다.'), Problem_Content(question='난잡한 코드를 리팩토링하여 메소드 이해 시간을 75% 단축하셨다고 했는데, Rich Domain Model, GoF 디자인 패턴 적용 등 어떤 구체적인 기법들을 사용하셨고, 그 효과는 어떻게 측정하셨나요?', explanation='코드 품질 개선에 대한 지원자님의 접근 방식과 실제 적용 사례, 그리고 그 효과를 측정하는 방법을 이해하기 위함입니다.'), Problem_Content(question='"주변시위 Now" 프로젝트를 진행하면서 겪었던 가장 큰 어려움은 무엇이었고, 그 어려움을 어떻게 극복하셨으며, 이 경험을 통해 무엇을 배우셨나요?', explanation='프로젝트를 통해 얻은 교훈과 이를 바탕으로 한 지원자님의 성장 가능성 및 문제 해결 능력을 확인하기 위함입니다.')]}, 
        {'problem_type': 'tech', 'content': [Problem_Content(question='K6를 사용하여 성능 테스트를 진행하셨는데, 구체적으로 어떤 지표들을 모니터링하셨고 병목 지점을 어떻게 식별하셨는지 설명해주시겠어요? 또한, Hikaricp Connection Pool과 Tomcat Thread Pool Size를 조정한 경험이 있으신데, 이 두 설정이 시스템 성능에 어떤 영향을 미치는지, 그리고 적정 값을 찾기 위해 어떤 과정을 거치셨는지 궁금합니다.', explanation='후보자님의 이력서에서 K6를 활용한 성능 테스트와 Hikaricp, Tomcat Thread Pool, Redis Cache 적용을 통한 인상적인 성능 개선 경험이 돋보였습니다. 실제 서비스 환경에서 발생할 수 있는 부하 상황에 대한 이해와 문제 해결 능력을 심층적으로 파악하고자 합니다.'), Problem_Content(question='제시된 아키텍처에서 Spring 서버를 Auto Scaling Group으로 구성하신 이유와, 실제 서비스 부하가 증가했을 때 Auto Scaling이 어떻게 동작하여 안정성을 확보할 수 있었는지 설명해주세요. 또한, RDS와 ElastiCache(ValKey)를 함께 사용하셨는데, 각 데이터베이스의 역할과 데이터를 분리하여 사용한 이유, 그리고 데이터 일관성 유지를 위한 전략이 있으셨는지 궁금합니다.', explanation='제시된 아키텍처에서 Spring 서버를 Auto Scaling Group으로 구성하고 RDS와 ElastiCache(ValKey)를 함께 사용하신 점은 확장성과 데이터 관리 전략에 대한 깊은 이해를 보여줍니다. 이에 대한 구체적인 설계 의도와 운영 경험을 듣고 싶습니다.'), Problem_Content(question="이력서에 'Rich Domain Model'과 'GoF Design Pattern'을 적용하여 코드 결합도 및 가독성을 개선하셨다고 기재되어 있습니다. 구체적으로 어떤 패턴들을 적용하셨고, 그로 인해 얻은 가장 큰 이점은 무엇이었나요? 또한, 'Layer 중심 모듈 구조를 Domain 중심 모듈 구조로 변경'하신 경험이 있으신데, 이 변경이 프로젝트의 유지보수성과 확장성에 어떤 긍정적인 영향을 미쳤는지 설명해주세요.", explanation="코드 결합도 및 가독성 개선을 위해 'Rich Domain Model'과 'GoF Design Pattern'을 적용하고 'Layer 중심 모듈 구조를 Domain 중심 모듈 구조로 변경'하신 경험은 소프트웨어 설계 원칙에 대한 깊은 이해를 보여줍니다. 이에 대한 구체적인 적용 사례와 그 효과를 듣고 싶습니다."), Problem_Content(question='Staging 서버 환경을 구축하고 Production 환경과 동일하게 Nginx에 SSL 인증서를 적용하고 Docker를 사용하셨다고 했습니다. 이러한 Staging 환경 구축이 개발 및 배포 프로세스에 어떤 이점을 가져다주었으며, 특히 협업 효율성 측면에서 어떤 개선이 있었는지 궁금합니다. 또한, Jenkins를 사용하여 CI/CD 파이프라인을 구축하신 것으로 보이는데, Jenkins 파이프라인의 주요 단계와 자동화된 배포 과정에서 발생할 수 있는 문제점들을 어떻게 관리하셨는지 설명해주세요.', explanation='Staging 서버 환경 구축과 Jenkins를 활용한 CI/CD 파이프라인 구축 경험은 개발 프로세스 효율화 및 안정적인 배포 능력에 대한 이해를 보여줍니다. 특히 팀 리더로서 협업 효율성 증대에 기여한 점을 높이 평가합니다.')]}, {'problem_type': 'cowork', 'content': [Problem_Content(question='"주변시위 Now" 프로젝트에서 팀 리더로서 팀원들과의 협업을 어떻게 이끌어 나갔으며, Staging 환경 구축이 팀의 협업 효율성에 어떤 긍정적인 영향을 주었는지 구체적으로 설명해주세요.', explanation='팀 리더로서 Staging 환경 구축 및 코드 가독성 개선을 통해 협업 효율성을 높인 경험이 있습니다. 팀원들과의 협업 방식과 리더십에 대해 질문합니다.'), Problem_Content(question='"동료가 이해하기 어려운 난잡한 코드" 문제를 해결하는 과정에서 팀원들과 어떻게 소통하고 협의했는지, 그리고 코드 리뷰는 어떤 방식으로 진행되었는지 구체적인 경험을 공유해주세요.', explanation='동료가 이해하기 어려운 코드 문제를 해결하여 메소드 이해 시간을 단축시킨 경험이 있습니다. 이 과정에서 팀원들과의 소통 방식과 코드 리뷰 경험에 대해 질문합니다.'), Problem_Content(question="'실시간 시위 응원하기' 기능의 성능 최적화 과정에서 K6를 활용하여 병목을 식별하고 개선하셨는데, 이 과정에서 팀원들과의 역할 분담은 어떻게 이루어졌으며, 기술적인 문제 해결을 위해 팀원들과 어떻게 협력했는지 설명해주세요.", explanation='성능 최적화 과정에서 K6를 활용하여 병목을 식별하고 개선했습니다. 이 과정에서 팀원들과의 역할 분담 및 기술적인 문제 해결을 위한 협력 방식에 대해 질문합니다.')]}]
    """

    delete_docs_by(key="metadata.user_id", value=state.user_id, collection_name=personalized_problems_collection_name)

    problem_docs = []
    for problems_with_type in state.problems:
        for problem_content in problems_with_type['content']:
            problem_doc = Document(
                page_content=f"question:{problem_content.question}\nexplanation:{problem_content.explanation}",
                metadata={
                    "problem_type": problems_with_type['problem_type'],
                    "user_id": state.user_id,
                    "api_version": state.api_version,
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

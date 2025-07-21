import getpass
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointIdsList, \
  PayloadSchemaType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings


if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
apply_docs_collection_name = os.getenv("APPLY_DOCS_COLLECTION_NAME")
personalized_problems_collection_name = os.getenv("PERSONALIZED_PROBLEMS_COLLECTION_NAME")

"""
- QdrantClient는 직접적인 종속성(Direct SDK)임.
- QdrantClient는 정말 필수적일 때만 사용하는 것이 좋다. 예를 들어, 컬렉션 생성, 페이로드 인덱스 생성 등.
- 그 외에는 vendor 비종속성 코드를 사용하는 것이 좋다.
"""
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# GoogleGenerativeAIEmbeddings에는 큰 문제가 있음. 
# 내부적으로 grpc 통신을 한다는데, 이거땜에 비동기로 여겨짐. 이거땜에 모든 코드를 전부 비동기로 변경해야 함. 하지만 잘 적용도 안됨!! event loop error!!
# embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ensure_collection_exists(collection_name: str, vector_size: int = 1536):
  """컬렉션이 존재하지 않으면 생성하고 필요한 인덱스를 설정합니다."""
  try:
    client.get_collection(collection_name)
  except Exception as e:
    client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_payload_index(
      collection_name=collection_name,
      field_name="metadata.user_id",
      field_schema=PayloadSchemaType.KEYWORD
    )

# 컬렉션들 초기화
ensure_collection_exists(apply_docs_collection_name)
ensure_collection_exists(personalized_problems_collection_name)


def create_vector_store(collection_name: str) -> QdrantVectorStore:
  """Factory function to create QdrantVectorStore instances."""
  return QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
    content_payload_key="page_content",
    metadata_payload_key="metadata",
  )

apply_docs_vector_store = create_vector_store(apply_docs_collection_name)
personalized_problems_vector_store = create_vector_store(personalized_problems_collection_name)

"""
  - as_retriever()가 리턴하는 `VectorStoreRetriever`가 바로 완전한 비종속성 코드임.
  - 따라서 아래와 같이 문서 검색 예시를 추상화하는 것이 좋다.
"""
def get_retriever_for_user(user_id: str) -> VectorStoreRetriever:
  return apply_docs_vector_store.as_retriever(search_type="similarity", search_kwargs={
  "k": 5,
  "filter": Filter(
    must=[
      FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
    ]
  )
})

def get_filter_condition(key: str, value: str) -> Filter:
  return Filter(
    must=[
      FieldCondition(key=key, match=MatchValue(value=value))
    ]
  )

def delete_docs_by(key: str, value: str, collection_name: str = apply_docs_collection_name):
  filter_condition = get_filter_condition(key, value)
  scroll_result = client.scroll(
    collection_name=collection_name,
    scroll_filter=filter_condition,
    with_payload=False,
    with_vectors=False,
    limit=30,
  )

  points = scroll_result[0]
  ids_to_delete = [point.id for point in points]

  if ids_to_delete:
    client.delete(
      collection_name=collection_name,
      points_selector=PointIdsList(points=ids_to_delete)
    )
    return True
  return False
import getpass
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointIdsList, \
  PayloadSchemaType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
apply_docs_collection_name = os.getenv("APPLY_DOCS_COLLECTION_NAME")
embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")

"""
- QdrantClient는 직접적인 종속성(Direct SDK)임.
- QdrantClient는 정말 필수적일 때만 사용하는 것이 좋다. 예를 들어, 컬렉션 생성, 페이로드 인덱스 생성 등.
- 그 외에는 vendor 비종속성 코드를 사용하는 것이 좋다.
"""
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

try:
  info = client.get_collection(apply_docs_collection_name)
except Exception as e:
  client.create_collection(
    collection_name=apply_docs_collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
  )

client.create_payload_index(
  collection_name=apply_docs_collection_name,
  field_name="metadata.user_service_id",
  field_schema=PayloadSchemaType.KEYWORD
)

"""
  - QdrantVectorStore는 직접적인 종속성(Direct SDK, QdrantClient)를 위한 표준화된 어댑터(Standardized Adapter)임.
  - vector_store를 초기화하는 코드. 반드시 필요함! 구성요소를 잘 살펴볼 것.
"""
vector_store = QdrantVectorStore(
    client=client,
    collection_name=apply_docs_collection_name,
    embedding=embeddings,
    content_payload_key="page_content",
    metadata_payload_key="metadata",
)


"""
  - as_retriever()가 리턴하는 `VectorStoreRetriever`가 바로 완전한 비종속성 코드임.
  - 따라서 아래와 같이 문서 검색 예시를 추상화하는 것이 좋다.
"""
def get_retriever_for_user(user_service_id: str) -> VectorStoreRetriever:
  return vector_store.as_retriever(search_type="similarity", search_kwargs={
  "k": 5,
  "filter": Filter(
    must=[
      FieldCondition(key="metadata.user_service_id", match=MatchValue(value=user_service_id))
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
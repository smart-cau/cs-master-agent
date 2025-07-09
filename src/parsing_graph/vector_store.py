import getpass
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointIdsList, \
  PayloadSchemaType
from langchain_google_genai import GoogleGenerativeAIEmbeddings


if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
apply_docs_collection_name = os.getenv("APPLY_DOCS_COLLECTION_NAME")
embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL")



client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

try:
  client.get_collection(apply_docs_collection_name)
except Exception as e:
  client.create_collection(
    collection_name=apply_docs_collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
  )

client.create_payload_index(
  collection_name=apply_docs_collection_name,
  field_name="metadata.user_id",
  field_schema=PayloadSchemaType.KEYWORD
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=apply_docs_collection_name,
    embedding=embeddings,
)

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

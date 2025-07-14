# Qdrant Use Manul

## Filter(공식문서[https://qdrant.tech/documentation/concepts/filtering/])

- Filter는 사용하기 전에 반드시 index를 생성해야 합니다.
- 생성 방법은 아래와 같습니다.

```python
client.create_payload_index(
  collection_name=apply_docs_collection_name,
  field_name="metadata.user_id",
  field_schema=PayloadSchemaType.KEYWORD
)
```

-----

### Qdrant 종속 기능, 언제 괜찮을까?

결론부터 말씀드리면, **모든 코드를 종속성 없게 만들 수는 없으며, 그럴 필요도 없습니다.** 중요한 것은 **어떤 부분이 종속성을 가져도 괜찮은지, 어떤 부분이 추상화되어야 하는지를 명확히 구분**하는 것입니다.

보내주신 코드는 이미 상당 부분 좋은 관행을 따르고 있지만, 더 명확하게 분리할 수 있습니다.

이를 판단하는 기준은 **"이 기능이 대부분의 벡터 DB에 공통적인가, 아니면 Qdrant 고유의 관리/최적화 기능인가?"** 입니다.

#### 1. LangChain 추상 계층을 사용해야 할 때 (종속성 없는 코드) 🧩

이 영역은 **애플리케이션의 핵심 로직**에 해당하며, 다른 벡터 DB로 교체하더라도 변하지 않아야 하는 부분입니다.

  * **해당 작업**:
      * **문서 추가**: `vector_store.add_documents()`
      * **문서 검색**: `retriever.get_relevant_documents()`
      * **ID 기반 문서 삭제**: `vector_store.delete(ids=...)`
  * **왜?**: 이 기능들은 LangChain의 `VectorStore` 클래스가 표준으로 정의한 핵심 기능입니다. Qdrant, FAISS, Chroma 등 대부분의 벡터 DB가 이 인터페이스를 지원하므로, 이 코드는 DB를 교체해도 안전합니다.

#### 2. `QdrantClient`를 직접 사용해야 할 때 (종속적인 코드) ⚙️

이 영역은 **데이터베이스 자체를 관리, 설정, 최적화**하거나 Qdrant만이 제공하는 **고급 기능을 활용**하는 부분입니다.

  * **해당 작업**:
      * **컬렉션 생성/확인**: `client.get_collection()`, `client.create_collection()`
      * **페이로드 인덱스 생성**: `client.create_payload_index()`
      * **메타데이터 필터를 이용한 고급 삭제/수정**: `client.scroll()`, `client.delete()` (사용자 정의 `delete_docs_by` 함수)
  * **왜?**:
      * **초기 설정(Setup)**: 컬렉션의 벡터 차원(`size`), 거리 계산 방식(`distance`), 페이로드 인덱스 등은 DB를 처음 설정할 때만 필요한 **관리자 수준의 작업**입니다. 이는 LangGraph의 런타임 로직이 아니며, Qdrant에 맞춰 설정하는 것이 당연합니다.
      * **고유 기능 활용**: 작성하신 `delete_docs_by` 함수가 아주 좋은 예입니다. LangChain의 기본 `vector_store.delete()`는 보통 문서의 고유 ID 목록을 받습니다. 하지만 **"특정 유저 ID를 가진 모든 문서를 삭제"** 와 같이 메타데이터 조건을 이용해 삭제하는 기능은 Qdrant의 `scroll`과 `filter`를 활용해야 효율적입니다. 이런 고급 기능은 `QdrantClient`를 직접 사용하는 것이 맞습니다.

-----

### 코드 리팩토링 제안

현재 코드를 역할에 따라 분리하면 종속성을 훨씬 더 깔끔하게 관리할 수 있습니다.

#### 1\. **DB 설정 코드를 별도 함수/스크립트로 분리**

컬렉션을 만들고 인덱스를 생성하는 코드는 애플리케이션 실행 시 매번 필요한 로직이 아닙니다. 별도의 초기화 스크립트나 함수로 분리하세요.

```python
# qdrant_setup.py

def setup_qdrant_collection(client, collection_name, embedding_size=768):
    """Qdrant 컬렉션과 인덱스를 초기 설정하는 함수"""
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
        # 필요한 인덱스도 여기서 생성
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.user_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print("Collection setup complete.")

# 메인 애플리케이션 시작 시 한 번만 호출
# setup_qdrant_collection(client, apply_docs_collection_name)
```

#### 2\. **Qdrant 전용 유틸리티 함수 분리**

`delete_docs_by`와 같이 Qdrant 클라이언트를 직접 사용하는 함수들은 `qdrant_utils.py`와 같은 파일에 모아두어 "이 부분은 Qdrant 종속적이다"라는 것을 명확히 합니다.

```python
# qdrant_utils.py

def delete_docs_by_metadata(client, collection_name: str, key: str, value: str):
    """메타데이터 필터를 사용해 Qdrant에서 문서를 삭제하는 유틸리티"""
    filter_condition = Filter(must=[FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))])
    
    # ... client.scroll 및 client.delete 로직 ...
    print(f"Deleted documents where {key} = {value}")
    return True
```

#### 3\. **LangGraph 노드 로직은 최대한 추상화 유지**

이제 LangGraph의 상태(State)나 노드(Node)에서는 추상화된 `vector_store`와 `retriever`를 사용하고, 필요할 때만 분리해둔 유틸리티 함수를 호출합니다.

```python
# graph_nodes.py
# from qdrant_utils import delete_docs_by_metadata

# ... VectorStore 및 Retriever 설정 ...

def add_documents_node(state):
    # LangChain의 표준 인터페이스 사용 (GOOD ✅)
    vector_store.add_documents(documents=state["new_docs"])
    return {"status": "Documents added"}

def delete_user_documents_node(state):
    user_id = state["user_id_to_delete"]
    
    # Qdrant 종속 유틸리티를 명시적으로 호출 (GOOD ✅)
    # 왜냐하면 '메타데이터로 삭제'는 고급 기능이기 때문
    delete_docs_by_metadata(
        client=client, 
        collection_name=apply_docs_collection_name,
        key="user_id",
        value=user_id
    )
    return {"status": f"Documents for user {user_id} deleted"}
```

-----

### 최종 요약

| 구분 | 언제 사용하나요? | 예시 코드 |
| :--- | :--- | :--- |
| **LangChain 추상화** 🧩 | 문서 추가, 검색 등 **일반적인 애플리케이션 로직** | `vector_store.add_documents()`, `retriever.get_relevant_documents()` |
| **`QdrantClient` 직접 사용** ⚙️ | DB 초기 설정, 인덱싱, **DB 고유의 고급 기능 활용** | `client.create_collection()`, `client.create_payload_index()`, 메타데이터 필터 기반 삭제 |

이렇게 역할을 분리하면, 나중에 DB를 교체해야 할 때 `qdrant_setup.py`와 `qdrant_utils.py`만 새로 작성하면 되므로, LangGraph의 핵심 로직은 전혀 건드릴 필요가 없어집니다. 이것이 바로 프레임워크를 사용해 종속성을 효과적으로 관리하는 방법입니다.


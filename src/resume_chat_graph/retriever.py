from langchain_core.vectorstores import VectorStoreRetriever

from constants.vector_store import vector_store, get_filter_condition


def get_retriever_for_user(user_id: str) -> VectorStoreRetriever:
    """
    Creates a retriever for a specific user, filtering by user_id in the metadata.
    Args:
        user_id (str): The ID of the user whose documents should be retrieved.
    Returns:
        VectorStoreRetriever: A retriever configured to fetch documents for the specified user.
    """

    search_kwargs = {
        "filter": get_filter_condition(key="metadata.user_id", value=user_id),
        "k": 10,
    }
    return vector_store.as_retriever(search_kwargs=search_kwargs)
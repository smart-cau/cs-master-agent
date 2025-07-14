# LangGraph Usage Guide

## Tools

- tool은 `@tool` decorator + ToolNode 방식으로 구현한다.
- 위와 같은 방식으로 구현해야 langSmith에서 추적할 수 있다.
- 이 때, 반드시 `name`, `description`을 명시해야 한다. 그래야 llm이 tool을 사용할 때 정확한 툴을 사용할 수 있다.
- 예시:

```python
@tool(name="retriever_tool", description="Searches and returns excerpts from the user's resume and career documents. Use it to answer questions about the user's experience, projects, and skills.")
def get_retriever_tool(query: str):
    """
    Searches and returns excerpts from the user's resume and career documents. Use it to answer questions about the user's experience, projects, and skills.
    """
```
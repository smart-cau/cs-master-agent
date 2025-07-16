"""State management for the retrieval graph.

This module defines the state structures and reduction functions used in the
retrieval graph. It includes definitions for document indexing, retrieval,
and conversation management.

Classes:
    RetrievalState: Represents the state for document retrieval operations.
    ConversationState: Represents the state of the ongoing conversation.

The module also includes type definitions and utility functions to support
these state management operations.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


#############################  Agent State  ###################################


# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.
@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""


# This is the primary state of your agent, where you can store any information


def add_queries(existing: Sequence[str], new: Sequence[str]) -> Sequence[str]:
    """Combine existing queries with new queries.

    Args:
        existing (Sequence[str]): The current list of queries in the state.
        new (Sequence[str]): The new queries to be added.

    Returns:
        Sequence[str]: A new list containing all queries from both input sequences.
    """
    return list(existing) + list(new)


@dataclass(kw_only=True)
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    user_id: str = field(default="b4a80540-b2f8-4b3a-abaf-bed3ecf99f4a")
    """The user id of the user."""

    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.

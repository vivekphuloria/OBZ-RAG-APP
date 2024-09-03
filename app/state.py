from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Represents the state of the  graph.

    Attributes:
        question: question
        generation: LLM generation
        classification: Classification of User Query
        documents: list of documents
        messages: List of All Messages
    """

    question: str
    classification: List[str]
    documents: List[str]
    messages:  Annotated[list, add_messages]

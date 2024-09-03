from typing import Any, Dict

from app.state import GraphState
from app.chains import get_query_classification_chain, get_generation_chain

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from .consts import LLM, LLM_GEN

load_dotenv()

from .consts import PINCONE_VECTORSTORE_INDEX_NAME, METADATA_TAGS
from .consts import  EMBEDDING_MODEL, LLM


## Defining Nodes

# May not use the user input node
def human_node(state: GraphState)-> Dict[str, Any]:
    return {'messages':HumanMessage(state['question'])} ## Placeholder node; will have changes made from the front-end app
    
 
def classify_node(state: GraphState) -> Dict[str, Any]:
    clf_chain = get_query_classification_chain(llm=LLM)
    clf = clf_chain.invoke(state['question'])['classification']
    return {'classification':clf, 'question': state['question']}


def retrive_node(state: GraphState) -> Dict[str, Any] :
    vectorstore = PineconeVectorStore(
        index_name=PINCONE_VECTORSTORE_INDEX_NAME,
        embedding=EMBEDDING_MODEL)
    
    # For retrival, only metadata_tags can be used. Only selecting those from the classification
    metadata_tag_filters = list(set(state['classification']).intersection(set(METADATA_TAGS)))

    retriver = vectorstore.as_retriever(
        search_kwargs= {
            "filter": {"clf": {"$in":metadata_tag_filters}},
            "k": 10})
    docs = retriver.invoke(state['question'])
    return {'documents':docs,  "question": state['question'], 'classification':state['classification']}



def generation_node(state: GraphState) -> Dict[str, Any]:
    generation_chain = get_generation_chain(LLM_GEN)
    
    # Only pass history if the user requested something relevant to it, else skip 
    history = state['messages'][-10:] if 'Previous' in state['classification'] else []


    response = generation_chain.invoke({
        "context" : state['documents'], 
        "question": state['question'],
        "history": history
        })
    # Additional AIMessage conversion is added to convey this information to the streamlit app later 
    return {"messages": AIMessage(response) }

def search_node(state: GraphState) -> Dict[str, Any]:
    response = "Functionality for search to be added"
    return {"messages": AIMessage(response) }


def invalid_node(state: GraphState) -> Dict[str, Any]:
    response = f"""
The chatbot can only answer questions about the Person's professional attributes- {METADATA_TAGS} 
Please ask questions related to these.""".strip()

    # Additional AIMessage conversion is added to convey this information to the streamlit app later 
    return {"messages": AIMessage(response) }



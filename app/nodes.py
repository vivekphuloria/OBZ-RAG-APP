from typing import Any, Dict

from app.state import GraphState
from app.chains import get_person_name_chain, get_query_classification_chain, get_generation_chain, get_search_generation_chain

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from .consts import LLM, LLM_GEN, EMBEDDING_MODEL
from .consts import PINCONE_VECTORSTORE_INDEX_NAME, METADATA_TAGS, RETRIVER_FETCH_K, LIST_PERSON_NAME

load_dotenv()



## Defining Nodes

## Placeholder node; will have changes made from the front-end app
def human_node(state: GraphState)-> Dict[str, Any]:
    return {'messages':HumanMessage(state['question'])} 
    
 
def person_name_node(state: GraphState)-> Dict[str, Any]:
    chain = get_person_name_chain(LLM)
    inferred_person_name = chain.invoke(state['question'])
    
    # If the person name exists from a previous run, use that  
    if (inferred_person_name=='') and ('person_name' in state) and (state['person_name']!=''):
        inferred_person_name = state['person_name']

    return {'person_name':inferred_person_name}


def classify_node(state: GraphState) -> Dict[str, Any]:
    clf_chain = get_query_classification_chain(llm=LLM)
    clf = clf_chain.invoke(state['question'])['classification']
    return {'classification':clf, 'question': state['question']}

# No LLM computation in this node. This is just a placeholder so that classifier and person_name can be held in parallel
def router_node(state: GraphState) -> Dict[str, Any]:
    if state['classification'] in METADATA_TAGS and state['person_name']=='':
        # We get the same response both in case when there's no name and when there's multiple names. Has to be fixed 
        response = f"This chatbot can only answer questions about the following people:{LIST_PERSON_NAME}.\nPlease mention the person name in your queries."
    elif ('Personal' in state['classification']) or ('Others' in state['classification']):
        response = f"The chatbot can only answer questions about the Person's professional attributes- {METADATA_TAGS}. \nPlease limit your questions to these topics."
    else:
        response='' # Will not be used elsewhere 
    return {'hardcoded_response': response}





def retrive_node(state: GraphState) -> Dict[str, Any] :
    vectorstore = PineconeVectorStore(
        index_name=PINCONE_VECTORSTORE_INDEX_NAME,
        embedding=EMBEDDING_MODEL)
    
    # For retrival, only metadata_tags can be used. Only selecting those from the classification
    metadata_tag_filters = list(set(state['classification']).intersection(set(METADATA_TAGS)))
    full_metadata_filter = { 
        "$and": 
        [
            {"clf": {"$in":metadata_tag_filters}},
            { "person_name": state['person_name'] }
      ] 
     }
    retriver = vectorstore.as_retriever(
        search_kwargs= {
            "filter": full_metadata_filter,
            "k": RETRIVER_FETCH_K})
    docs = retriver.invoke(state['question'])
    return {'documents':docs}



def generation_node(state: GraphState) -> Dict[str, Any]:
    chain_func = get_search_generation_chain if 'Unrelated' in state['classification'] else get_generation_chain
    generation_chain = chain_func(LLM_GEN)
    
    # Only pass history if the user requested something relevant to it, else skip 
    history = state['messages'][-10:] if 'Previous' in state['classification'] else []


    response = generation_chain.invoke({
        "documents" : state['documents'], 
        "question": state['question'],
        "history": history
        })
    # Additional AIMessage conversion is added to convey this information to the streamlit app later 
    return {"messages": AIMessage(response) }

def search_node(state: GraphState) -> Dict[str, Any]:
    tool = TavilySearchResults(max_results=2)
    search_response =  tool.invoke(state['question'])

    return {"documents": search_response }


def invalid_node(state: GraphState) -> Dict[str, Any]:
    response = state['hardcoded_response']

    # Additional AIMessage conversion is added to convey this information to the streamlit app later 
    return {"messages": AIMessage(response) }



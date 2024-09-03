from .state import GraphState
from .consts import RETRIVER_NODE, INVALID_NODE, SEARCH_NODE, GENERATION_NODE, METADATA_TAGS

## Defining Conditional Edges
def func_clf_router(state: GraphState) -> str:
    s_clf = set(state['classification']) 
    
    # If the query requires any information about the metadata
    if len(s_clf.intersection(set(METADATA_TAGS)))>0:
        return 'RETRIVER'

    # If question only requests information from previous, without any other context
    elif state['classification'] == ['Previous']:
        return "GENERATION"
    
    # Unrelated queries should go to search
    elif "Unrelated" in state['classification']:
        return "SEARCH"

    # Will be only called in case the question is personal or something else about the user
    else:
        return "INVALID"

    
d_clf_router = {
    "RETRIVER": RETRIVER_NODE,
    "GENERATION": GENERATION_NODE, 
    "SEARCH":SEARCH_NODE,
    "INVALID": INVALID_NODE}
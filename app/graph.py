from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import GraphState
from .nodes import human_node, classify_node, search_node, retrive_node, generation_node, invalid_node 
from .edges import func_clf_router, d_clf_router

from .consts import HUMAN_NODE, CLASSIFY_NODE, SEARCH_NODE, RETRIVER_NODE, GENERATION_NODE, INVALID_NODE


load_dotenv()

def get_graph(): 
    graph =  StateGraph(GraphState)

    graph.add_node(HUMAN_NODE, human_node)
    graph.add_node(CLASSIFY_NODE, classify_node)
    graph.add_node(SEARCH_NODE, search_node)
    graph.add_node(RETRIVER_NODE, retrive_node)
    graph.add_node(GENERATION_NODE, generation_node)
    graph.add_node(INVALID_NODE, invalid_node)
    
    graph.add_edge(START, HUMAN_NODE)
    graph.add_edge(HUMAN_NODE, CLASSIFY_NODE)
    graph.add_conditional_edges(CLASSIFY_NODE, func_clf_router, d_clf_router) ## Classification node can route to unrelated or retriver node

    graph.add_edge(RETRIVER_NODE, GENERATION_NODE)
    graph.add_edge(SEARCH_NODE, GENERATION_NODE)
    graph.add_edge(GENERATION_NODE, HUMAN_NODE) ## Adding Curcular References
    
    graph.add_edge(INVALID_NODE , HUMAN_NODE) ## Will shift it eventually to tavily and others 
    
    memory = SqliteSaver.from_conn_string('obz-rag-app.sqlite')


    app = graph.compile(checkpointer=memory, interrupt_before=[HUMAN_NODE])
    
    # Draw graph onto file; Optional  
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")

    return app

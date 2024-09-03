from langchain.prompts import ChatPromptTemplate 
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from .consts import ALL_TAG_DESCRIPTION, METADATA_TAGS


## Structured Output for Query Classification and RAG-Document 
class InputClassification(BaseModel):
    """ 
    Classify the input into one or more of the categories
    """ 
    classification : list[str] = Field(description = "List of classifications the input belongs to")

# some local functions to be used to create prompts for the Classificaiton Chains
get_description_string = lambda d_desc: '\n'.join([f"- {k} : {v}" for k,v in  d_desc.items()])
get_tags_list_string = lambda d_desc: ', '.join([f'"{k}"' for k in d_desc.keys()])



def get_query_classification_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ('system', f"""
         You are part of a QnA app for a person's profile. 
         You will be given a question. Your task is to classify it into one or more of the following classifications: {get_tags_list_string(ALL_TAG_DESCRIPTION)}
         Use the followig following instructions to classify the question
         {get_description_string(ALL_TAG_DESCRIPTION)}
         ENSURE that the response should only be a single element or a comma separated list from the above options. No other words can be used
         """),
         ("human","{input}")
    ])
    
    llm_st = llm.with_structured_output(InputClassification)
    query_clf_chain = (
        prompt |
        llm_st |
        RunnableLambda(lambda x: x.dict())
    )
    return query_clf_chain


# Used in the ingestion process
def get_chunk_classification_chain(llm):
    metadata_tag_desc = {k:ALL_TAG_DESCRIPTION[k] for k in METADATA_TAGS}
    instructions = get_description_string(metadata_tag_desc).replace("question","information")
    
    prompt = ChatPromptTemplate.from_messages([
        ('system', f"""
         You are analysing data about a peron's profile.
         You will be given a some information and your task is to classify it into one or more of the following classifications: {get_tags_list_string(metadata_tag_desc)}
         Use the followig following instructions to classify the information
         {instructions}
         ENSURE that the response should only be a single element or a comma separated list from the above options. No other words can be used
         """),
         ("human","{input}")
    ])
    
    llm_st = llm.with_structured_output(InputClassification)
    query_clf_chain = (
        prompt |
        llm_st |
        RunnableLambda(lambda x: x.dict())
    )
    return query_clf_chain


def get_generation_chain(llm):
    prompt =ChatPromptTemplate.from_messages([
        ("human","""
         You are an assistant for a question-answering chatbot. 
         You will be provided the user's current question, retrieved context, and the Chat history (previous messages in the chat). 
         Use only the retrived context or the chat history to answer the current question. Do not use any other information.
         If the information required for the answer is not present in the mentioned sources, respond with "Information not present in context".
         Your answer should be markdown friendly - highlight the proper nounds and use bullets or other emphasis whereever neccessary.
         Question: {question} 
         Context: {context}
         Chat History: {history}
         Answer: """)
    ])
    chain =  prompt | llm | StrOutputParser()
    return chain



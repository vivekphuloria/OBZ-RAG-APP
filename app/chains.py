from langchain.prompts import ChatPromptTemplate 
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from fuzzywuzzy import process
from .consts import ALL_TAG_DESCRIPTION, METADATA_TAGS, LIST_PERSON_NAME


## Structured Output for Query Classification and RAG-Document 
class InputClassification(BaseModel):
    """ 
    Classify the input into one or more of the categories
    """ 
    classification : list[str] = Field(description = "List of classifications the input belongs to")

# some local functions to be used to create prompts for the Classificaiton Chains
get_description_string = lambda d_desc: '\n'.join([f"- {k} : {v}" for k,v in  d_desc.items()])
get_tags_list_string = lambda d_desc: ', '.join([f'"{k}"' for k in d_desc.keys()])


def get_person_name_chain(llm):
    ## Function for fuzzy matching of name to list of names
    def get_match(query):
        match = process.extractOne(query, LIST_PERSON_NAME, score_cutoff=60)
        return match[0] if match else ''

    prompt = ChatPromptTemplate.from_messages([
    ('system', f"""You will be given a question about a person. 
     You have to extract the name of the person. If you don't think there's any person within the question, return an empty string('')
     Respond with only the person name and nothing else."""),
        ("human","{input}")
        ])
    chain = prompt | llm | StrOutputParser() | RunnableLambda(get_match)
    return chain


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
         Context: {documents}
         Chat History: {history}
         Answer: """)
    ])
    chain =  prompt | llm | StrOutputParser()
    return chain

def get_search_generation_chain(llm):
    prompt =ChatPromptTemplate.from_messages([
        ("human","""
         You are an assistant for a question-answering chatbot. 
         You have been provided the user's current question, the list of top search results for the query, and may also be provided with the previous chat of the user.
         Answer the user's question based on the  search results and if there is any relevant information in the chat history.
         If all these sources do not answer the question, respond that you don't know the answer to this question.
         Your answer should be markdown friendly - highlight the proper nounds and use bullets or other emphasis whereever neccessary.

         Question: {question} 
         Search Results: {documents}
         Chat History: {history}
         
         After your answer, tell the user that the primary function of the chatbot was to answer question about a person's professsional journey, and from the next question they should try to stick to that theme.  
         """)])
    chain =  prompt | llm | StrOutputParser()
    return chain


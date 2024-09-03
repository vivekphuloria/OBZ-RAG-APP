from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

PERSON_NAME = "Vivek Phuloria"
PINCONE_VECTORSTORE_INDEX_NAME = 'obz-rag-app'
RETRIVER_FETCH_K = 10

HUMAN_NODE      = 'HUMAN_NODE'
CLASSIFY_NODE   = 'CLASSIFY_NODE'
SEARCH_NODE     = 'SEARCH_NODE'
INVALID_NODE  = 'UNRELATED_NODE'
RETRIVER_NODE   = 'RETRIVER_NODE'
GENERATION_NODE = 'GENERATION_NODE'


ALL_TAG_DESCRIPTION = {
    'Education' : "If the question is regarding the person's educational qualifications, degrees, professional certifications, courses etc.",
    'Workex' : 'If the question is regarding the companies they have worked for, their jobs, internships etc or regarding their projects or work experience',
    'Skills' : 'If the question is regarding any tool, technology or professional skill that they have expertise in',
    'Interests' : 'If the question is regarding their hobby, or non-professional interest',
    'Contact' : 'If the question is regarding contact information like mail, phone number or social media profiles of the person',
    'Awards' : 'If the question is regarding any awards, titles, recognition or competitions that the person has won',
    'Personal' : 'If the question is regarding their family, personal life, belief systems etc',
    'Others' : 'If the question refers is asking a query about the user but is not related to any of the above buckets ',
    'Previous' : 'If the question asks something about a previous responses, or mentions anything about the conversation uptill now',
    'Unrelated' : 'If the question is not about the individual or their profile at all'
}
METADATA_TAGS = ['Education', 'Workex', 'Skills', 'Interests', 'Personal','Contact','Awards']


EMBEDDING_MODEL = OpenAIEmbeddings()
LLM = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
LLM_4o = ChatOpenAI(model='gpt-4o', temperature=0)
LLM_GEN = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
LLM_GEN_4o = ChatOpenAI(model='gpt-4o', temperature=0.5)

# print(RETRIVER_NODE)
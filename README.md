# ReadMe

## Functionality

- System Architecture
- Evaluation
- Demonstration

# Overview

The app aims at creating a chatbot, which can be answer questions about an Individual. 

The app is powered by RAG (Retrieval Augmented Generation), wherein data about the person is ingested to a  vector-store, and queried based on the user query. 

# How to Set UP

To set the project up locally

- Clone the repository on your local system using `git clone`
- Install the prerequisite modules on your local by creating a new environment  and using `pip install -r requirements.txt`
- Create a `.env` file with the API secrets. This should contain
    - PINECONE_API_KEY
    - PINECONE_API_ENV
    - OPENAI_API_KEY
    - TAVILY_API_KEY = "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"
    
    Optional, For LangSmith Tracing
    
    - LANGCHAIN_API_KEY
    - LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com/)"
    - LANGCHAIN_TRACING_V2 = true ⇒ For Langsmith Tracing
    - LANGCHAIN_PROJECT

# Code Overview

### Document Tagging for RAG

An architectural decision taken for RAG was to tag the user-query and the chunks into what I thought are the different topics that could be - Education, Workex, Skills, Interests, Personal, Contact, Awards.    

During the ingestion process, each document chunk has been tagged into one or more of these categories using an LLM, and this information stored in the chunk metadata in the vecorDB.

During the  runtime, the user’s question asked by will also be tagged into one or more of these categories, and the retriever will apply metadata filters to only look for chunks with the relevant tags.  

Although, this approach adds an overhead of classifying the input during ingestion and run-time, but the pros of using this approach are

- Better retrieval - Given the embedding search’s scope is limited to only relevant documents of that tag, the chances of irrelevant documents is much lesser
- Flexible app design - We may want different types of questions to be answered in different ways - eg.
    - changing the tone, or highlight different types of information,
    - Restrict certain types of questions,  eg. not respond to personal queries about the person.
    - Redirect to different app-routes - eg: queries not related to user shouldn’t even go through the RAG pipeline
        - Scalability - If in the future, we wanted to connect the QnA system to other ERP systems like Attendance, Performance, Payroll etc or any other source of structured data.

During the runtime, the name of the person is also extracted from the query. Currenly there are only 2 people whose data exists on the plaform, but as this number increases, it'll be helpful to be able to identify whose information is being requested.

### Code Structure

The code-base contains the following directories and files

- 📄 .env → Contains API secrets
- 📄 st_app.py → Streamlit app that imports the graph from app.graph and renders it into a chatbot interface
- 📄 obz-rag-app.sqlite → Memory for storing the conversations through the graph states
- 📄 ingest.ipynb → Contains the different steps to ingest different files from rag_data. Have made a notebook and not a .py file, because this is an iterative process of testing different document loaders, splitters, and the tagging process.
- 📂 rag_data → Contains text-files, PDFs and HTMLs with the person’s details to be ingested for RAG
- 📂 app → app Backend
    - __init__.py → Indicate to python that this is a package
    - consts.py  → Contains constant values to be used in multiple places through the remaining app
    - chains.py  → Contains Langchain chains, used for classifying user queries and documents during ingestion, and generation chains for creating the final response.
    - state.py  → Contains the Graph State class, used as a reference for creating the nodes and by the final graph.py file
    - nodes.py → Creating the different nodes for the graph
    - nodes.py → Creating the different nodes for the graph
    - edges.py → Creating the conditional edges and routers for the graph
    - graph.py → Combining the nodes and edges to create the graph

### Ingestion

All the data to be ingested is stored in a folder. For the sake of convinience, a nomenclature of *<FirstName>-<LastName> <DocuementType>.<Extension>* was used to make it easier to tag documents, but a smarter method of inferring this information could also have been used for a production-ready system.

We iterate through all the files in the folder , pick a loader based on extension, split the documents, tag them and upload them to the VectorDB.

At each step, the results are observed to see, if they seem alright, or anything needs to be tweaked.  

For this application, simple loaders like *BSHTMLLoader*, *PyPDFLoader*, *TextLoader* were used given limited types and simple formats of documents. Advanced loaders like *Unstructured* or other data-type specific loaders may be used in production applications. 

Similartly, a simple *RecursiveCharacterTextSplitter* was used to split the documents, but better splitters like document type-specific splitters, Sematic splitter, or Agentic Splitters may be used for production-ready applications. 
However, tagging the ingested documents, reduces the sensitivity to proper chunking somewhat.

 

### Backend Flow

Langgraph has been used for orchestration. 

Langgraph allows creation of a control-flow graph with different nodes, each performing some operation, sharing a common state, whose value is transformed as the code passes through it.  

The graph state contains the user-question, it’s classificaitons, documents retrived(if any), and a list of messages. 

The code starts from the HUMAN_NODE, where the query is added through the streamlit front-end. This query goes to the CLASSIFICATION_NODE where an LLM chain identifies what is query about.  From here, the flow may now be redirected to different nodes based on it’s classification.

If query is about the person’s proffesional profile -  Education, Workex, Skills, Interests, Contact, Awards, it is redirected to the RETRIVER_NODE <which retrives data for the RAG>. 

I the query is about the person’s personal life - it is deemed invalid, and sent to the INVALID NODE, which returns a hard-coded string requesting the user to ask a valid query.

In case of the query not about the user, it is seen as a general topic, which goes to the SEARCH_NODE, which uses Tavily to search the internet for relevant information. 

If the flow has reached the GENERATION_NODE, it uses  the information provided from the RAG, Chat History, or the Search to respond to the user-query. 

Post the response, the flow is transferred back to the HUMAN_NODE where questions can be asked again.

 

![GRAPH](https://github.com/vivekphuloria/obz-rag-app/blob/main/graph.png)

To allow continuity, and allow human-in-the-loop, a SQLite based memory checkpointing was used.  

### Frontend

The front-end was created as a streamlit application. The older messages in a conversation were pulled from the SQLite memory and stored as session state variable. 

***The THREAD-ID implementation*** 

The Langgraph memory is indexed on a thread-id, and all messages in a conversation are stored against that thread-id.

A unique problem of managing the thread-id was encountered due to the way Streamlit fundamentally works.

Ideally a UUID should have been used to create a thread ID for a run, and create a new ID when the user consciously starts a new chat. 
But that was difficult to implement, given that Streamlit is otherwise stateless, and refreshes every-time any input is changed. A UUID cannot be assigned as a session-state variable because that also will be changed on every refresh. 
Hence there was no way of determining whether a refresh done automatically by streamlit app changing, or initiated by the user. 

A hacky approach of creating a thread-id based on current date-hour which is modifiable by the user was used. Any time the user re-starts the chat after some time, a new thread-id is created. The user can go back to the previous chat if they remember the thread-id of that conversation, and can manually edit the thread-id if they want to create a new chat within the same hour. 

This problem may not even arise if a different front-end is used, and even for Streamlit, a more elegant solution may be thought of. 

# Evaluation

RAGAS framework can be used for evaluation of the system. 

The following components required for assessing RAG

- Question → Asked by user
- Answer → Final response of the applicaiton
- Context → Retrieved context by the retrival pipeline
- Ground Truth → Best possible answers , Provided by humans

The metrics computed by RAGAS are 

- Faithfulness →
    - If generated answer is grounded in the context extracted by app. Checks hallucination of the model
- Answer Relevance →
    - if the answer is relevant to the original query
- Context Precision
    - If the context is relevant to the original question asked
- Context Recall
    - If the context retrieved is relevant to the ground truth

|  | Question | Answer | Context | Ground Truth |
| --- | --- | --- | --- | --- |
| Faithfullness |  | X | X |  |
| Relevance | X | X |  |  |
| Context Recall |  |  | X | X |
| Context Precision | X |  | X |  |

For implementing RAGAS, we need to create a set of questions to test our result and the write the ground truth for each, there are python libraries for implementing this.

Each metric conveys the health of a different aspect of the RAG framework, with different things that need to be fixed to improve them.

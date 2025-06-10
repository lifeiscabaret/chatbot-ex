import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

## 환경변수 읽어오기 ==============================================================
load_dotenv() 


## llm 함수 정의 ===================================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model) 
    return llm

## database 함수 정의 ================================================
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

    ## 임베딩 모델 지정text-embedding-3-large')
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_type=OPENAI_API_KEY
    )

    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'law' #변수에 저장

    ## 저장된 인덱스 가져오기
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )

## retrievalQA 함수 정의 ====================================================================
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## vector store에서 index 정보
    database = get_database()

    ## prompt
    # prompt = hub.pull('rlm/rag-prompt'), api_key=LANGCHAIN_API_KEY)
    prompt = hub.pull('lifeiscabaret/rag-prompt', api_key=LANGCHAIN_API_KEY)

    ## LLM 모델 지정
    llm = get_llm()

    def format_docs(docs):
        return'\n\n'.join(doc.page_content for doc in docs)

    qa_chain = ( #구성(설계)
        {
            'context' : database.as_retriever() | format_docs,
            'question' : RunnablePassthrough(),

        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain





## [AI Message 함수 정의] -------------------------------------------------------------
def get_ai_message(user_message):
    qa_chain = get_retrievalQA()
    ai_message = qa_chain.invoke(user_message)

    return ai_message

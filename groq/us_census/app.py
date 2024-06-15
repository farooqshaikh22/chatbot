import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

## loading groq api key

groq_api_key = os.environ["GROQ_API_KEY"]

st.title("ChatGroq with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question:{input}
    
    """
)

def vector_embeddings():
    
    if "vector" not in st.session_state:
        
  
        st.session_state.embeddings = OllamaEmbeddings()
        
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)
        
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        st.session_state.db = FAISS.from_documents(documents=st.session_state.documents,
                         embedding=st.session_state.embeddings)   
        
        

question = st.text_input("Enter your question from documents")

if st.button("Document Embedding"):   
    vector_embeddings()
    
document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.db.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)
    
if question: 
    response = retriever_chain.invoke({"input":question})
    st.write(response["answer"])
    
    
                            
    

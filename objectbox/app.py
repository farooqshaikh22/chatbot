import streamlit as st
import os
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_objectbox import ObjectBox
from dotenv import load_dotenv

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

st.title("ObjectBox VectorstoreDB with Llama3 Demo")

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
    if "vectors" not in st.session_state:
        #logger.info("Embedding process started.")
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        #logger.info("documents loaded.") 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        #logger.info("Documents split into chunks.")  
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.documents,st.session_state.embeddings,embedding_dimensions=768)
        #logger.info("Embeddings created and stored in ObjectBox.")


if st.button("Document Embeddings"):
    vector_embeddings()
    st.success("Documents have been embedded successfully!")
    #logger.info("Embedding process completed successfully.")
    
if "vectors" in st.session_state:
    input_prompt = st.text_input("Please Type a question")
    
    if input_prompt:  
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever,document_chain)
        response = retrieval_chain.invoke({"input":input_prompt})
        st.write(response["answer"])
        #logger.info("Response generated and displayed.")

else:
    st.warning("Please embed the documents first by clicking the 'Documents Embedding' button.")
    #logger.warning("User attempted to ask a question before embedding documents.")


    
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_objectbox.vectorstores import ObjectBox
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

st.title("ObjectBox VectorstoreDB with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embeddings():
    if "db" not in st.session_state:
        embeddings = OllamaEmbeddings()
        loader = PyPDFDirectoryLoader("./us_census")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs[:2])
        db = ObjectBox.from_documents(documents, embeddings, embedding_dimensions=768)
        
        st.session_state["db"] = db
        st.session_state["document_embeddings_created"] = True
        st.success("Documents have been embedded successfully!")
    else:
        st.warning("ObjectBox database already exists. Proceeding to the next step.")

if st.button("Document Embeddings"):
    vector_embeddings()

if "document_embeddings_created" in st.session_state:
    input_prompt = st.text_input("Please Type a question")

    if input_prompt:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state["db"].as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": input_prompt})
        st.write(response["answer"])
else:
    st.write("Please embed the documents first by clicking the 'Document Embeddings' button.")

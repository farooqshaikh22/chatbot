import streamlit as st
import os
from together import Together
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone
import pinecone

from dotenv import load_dotenv

load_dotenv()
pinecone_region = os.environ["PINECON_REGION"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
together_api_key = os.environ["TOGETHER_AI_API_KEY"]

together_ai = Together(api_key=together_api_key)

## loading document

loader = PyPDFDirectoryLoader("./medical")
docs = loader.load()
#print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

documents = text_splitter.split_documents(docs)

#print(len(documents))

embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#print(embeddings_model)


pinecone.init(api_key=pinecone_api_key,environment=pinecone_region)
index_name = "langchain-chatbot"

# Initialize Pinecone index and store embeddings

index = Pinecone.from_documents(documents=documents,embedding=embeddings_model,index_name=index_name)

print("Embedding stored in Pinecone Vector Database Successfully")

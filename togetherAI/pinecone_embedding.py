import streamlit as st
import os
from together import Together
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

from dotenv import load_dotenv

load_dotenv()

pinecone_region = os.getenv("PINECONE_REGION")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
together_api_key = os.getenv("TOGETHER_AI_API_KEY")

# Print environment variables to debug
print(f"Pinecone Region: {pinecone_region}")
print(f"Pinecone API Key: {pinecone_api_key[:5]}...")  # Only showing first few characters for security
print(f"Together AI API Key: {together_api_key[:5]}...")

# Loading documents
loader = PyPDFDirectoryLoader("./medical")
docs = loader.load()
print(f"Loaded {len(docs)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)
print(f"Split into {len(documents)} documents")

embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Pinecone client
print("Initializing Pinecone...")
try:
    pc = PineconeClient(api_key=pinecone_api_key)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Failed to initialize Pinecone: {e}")
    raise

index_name = "langchain-chatbot"

# Check if the index already exists
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Update this with the correct dimension for your embeddings model
        metric='cosine',  # or 'euclidean' or 'dotproduct' depending on your use case
        spec=ServerlessSpec(
            cloud='aws',
            region=pinecone_region
        )
    )
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# Store embeddings in Pinecone using Langchain's Pinecone class
print("Storing embeddings in Pinecone...")
# try:
#     index = LangchainPinecone.from_documents(documents=documents, embedding=embeddings_model, index_name=index_name)
#     print("Embeddings stored in Pinecone successfully.")
# except Exception as e:
#     print(f"Failed to store embeddings in Pinecone: {e}")
# Initialize Pinecone client
print("Initializing Pinecone...")
try:
    pc = PineconeClient(api_key=pinecone_api_key)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Failed to initialize Pinecone: {e}")
    raise

index_name = "langchain-chatbot"

index = LangchainPinecone.from_existing_index(index_name=index_name,embedding=embeddings_model)
  
q_ = "what is acne?"
r_ = index.similarity_search(q_,k=3)
c_ = ""

for match in r_:
    c_ += match.page_content + "\n"
print(r_)
print(c_)


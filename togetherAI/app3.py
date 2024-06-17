import streamlit as st
import os
from together import Together
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone as PineconeClient
from langchain.chains.combine_documents import create_stuff_document_chain
from langchain.chains.retrieval import create_retriever_chain

from dotenv import load_dotenv

load_dotenv()
pinecone_region = os.environ["PINECONE_REGION"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
together_api_key = os.environ["TOGETHER_AI_API_KEY"]

together_ai = Together(api_key=together_api_key)

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

index = LangchainPinecone.from_existing_index(index_name=index_name, embedding=embeddings_model)

# Create the document chain
document_chain = create_stuff_document_chain(
    loader=PyPDFDirectoryLoader("./your_document_directory"),
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
    embeddings_model=embeddings_model
)

# Create the retriever chain
retriever_chain = create_retriever_chain(
    retriever=index.as_retriever(search_kwargs={'k': 3}),
    document_chain=document_chain
)

# Streamlit application
st.title("Medical Chatbot with Together AI, Langchain, and Pinecone")

# Initialize session state for conversation history
if "requests" not in st.session_state:
    st.session_state["requests"] = []

if "responses" not in st.session_state:
    st.session_state["responses"] = ["How can I assist you!"]

# User input and Chatbot interaction
user_query = st.text_input("Enter Your Query...")
print("\n" + "="*50)
print(f"user_query : {user_query}")
print("="*50 + "\n")
if user_query:
    conversation = ""
    for i in range(len(st.session_state['responses'])):
        if len(st.session_state['requests']) > i:
            conversation += f"User: {st.session_state['requests'][i]}\n"
        if len(st.session_state['responses']) > i:
            conversation += f"Bot: {st.session_state['responses'][i]}\n"

    # Invoke the retriever chain with user input
    docs = retriever_chain.invoke({"input": user_query})
    print("\n" + "="*50)
    print(f"Docs : {docs}")
    print("="*50 + "\n")

    # Extract context from the documents
    context_list = [doc.page_content for doc in docs]
    print("\n" + "="*50)
    print(f"Context List : {context_list}")
    print("="*50 + "\n")
    context = "\n".join(context_list)
    print("\n" + "="*50)
    print(f"Context : {context}")
    print("="*50 + "\n")

    # Generate prompt for Together AI
    response = together_ai.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": context}]
    )
    bot_response = response.choices[0].message.content

    # Store the query and response in session state
    st.session_state["requests"].append(user_query)
    st.session_state["responses"].append(bot_response)

    # Display bot response
    st.write("Bot Response :")
    st.write(bot_response)

# Display conversation history
st.subheader("Conversation History:")
for i in range(len(st.session_state['responses'])):
    if len(st.session_state['requests']) > i:
        st.text(f"User: {st.session_state['requests'][i]}")
    if len(st.session_state['responses']) > i:
        st.text(f"Bot: {st.session_state['responses'][i]}")

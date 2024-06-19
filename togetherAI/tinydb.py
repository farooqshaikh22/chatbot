"""
Integrating TinyDB with Streamlit:
You can use TinyDB in a Streamlit application to manage session data, user inputs, or any other application-specific data. 
Here’s an example of how you might integrate TinyDB with a Streamlit app:
    
"""
import streamlit as st
from tinydb import TinyDB, Query

# Initialize TinyDB
db = TinyDB('db.json')

# Streamlit app title
st.title('TinyDB with Streamlit')

# Form to add a new user
with st.form('Add User'):
    name = st.text_input('Name')
    age = st.number_input('Age', min_value=0)
    city = st.text_input('City')
    submit = st.form_submit_button('Add User')

    if submit:
        db.insert({'name': name, 'age': age, 'city': city})
        st.success(f'Added {name} to the database!')

# Display users from the database
st.header('Users')
User = Query()
users = db.all()
for user in users:
    st.write(f"Name: {user['name']}, Age: {user['age']}, City: {user['city']}")

# Form to search for a user
with st.form('Search User'):
    search_name = st.text_input('Search by Name')
    search_submit = st.form_submit_button('Search')

    if search_submit:
        results = db.search(User.name == search_name)
        if results:
            for result in results:
                st.write(f"Name: {result['name']}, Age: {result['age']}, City: {result['city']}")
        else:
            st.warning(f'No users found with name {search_name}')




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
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()
pinecone_region = os.environ["PINECONE_REGION"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
together_api_key = os.environ["TOGETHER_API_KEY"]

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

    # Define the prompt template
    prompt_template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    qa_prompt = prompt.format(context=context, question=user_query)
    print(f"qa_prompt : {qa_prompt}")

    # Generate response using Together AI
    response = together_ai.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": qa_prompt}]
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



    """
    Store and retrieve user session data, which can help maintain state across multiple interactions or sessions.
    This includes tracking user preferences, previous queries, and responses.
    """
    
    from tinydb import TinyDB, Query

# Initialize the TinyDB
db = TinyDB('chatbot_db.json')
user_sessions = db.table('user_sessions')

# Function to save session data
def save_session(user_id, session_data):
    User = Query()
    user_sessions.upsert({'user_id': user_id, 'session_data': session_data}, User.user_id == user_id)

# Function to load session data
def load_session(user_id):
    User = Query()
    result = user_sessions.get(User.user_id == user_id)
    return result['session_data'] if result else None



feedback_table = db.table('feedback')

# Function to save feedback
def save_feedback(user_query, bot_response, feedback):
    feedback_table.insert({'query': user_query, 'response': bot_response, 'feedback': feedback})

# Example feedback collection
feedback = st.text_input("Provide your feedback on the response:")
if feedback:
    save_feedback(user_query, bot_response, feedback)
    st.write("Thank you for your feedback!")


from collections import Counter

# Function to get frequently asked questions
def get_frequent_queries():
    queries = [item['query'] for item in db.all()]
    return Counter(queries).most_common(10)

# Display frequent queries
st.write("Frequently Asked Questions:")
for query, count in get_frequent_queries():
    st.write(f"{query} - {count} times")


#"Maintain profiles for users, storing their preferences, interaction history, and any personalized settings."
user_profiles = db.table('user_profiles')

# Function to save user profile
def save_user_profile(user_id, profile_data):
    User = Query()
    user_profiles.upsert({'user_id': user_id, 'profile_data': profile_data}, User.user_id == user_id)

# Function to load user profile
def load_user_profile(user_id):
    User = Query()
    result = user_profiles.get(User.user_id == user_id)
    return result['profile_data'] if result else None

# Example usage
user_id = "user123"
profile_data = {"name": "John Doe", "preferences": {"language": "en"}}
save_user_profile(user_id, profile_data)
st.write("User profile saved!")






"""
Given that your senior mentioned storing some ID from Pinecone in TinyDB, it's likely related to maintaining a record of document metadata or a mapping between user queries and document IDs. This can help in tracking which documents are frequently accessed or for other analytical purposes.
Here's a way to integrate TinyDB for storing and managing IDs from Pinecone:
"""
import streamlit as st
import os
from together import Together
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from tinydb import TinyDB, Query
from dotenv import load_dotenv

load_dotenv()
pinecone_region = os.environ["PINECONE_REGION"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
together_api_key = os.environ["TOGETHER_AI_API_KEY"]

# Initialize Together AI
together_ai = Together(api_key=together_api_key)

# Initialize embeddings model
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

# Initialize TinyDB
db = TinyDB('pinecone_query_log.json')
query_log = db.table('query_log')

# Prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Initialize the retriever
retriever = index.as_retriever(search_kwargs={'k': 3})

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

    # Fetch relevant documents from Pinecone
    docs = retriever.get_relevant_documents(user_query)
    print("\n" + "="*50)
    print(f"Docs : {docs}")
    print("="*50 + "\n")

    # Store query and document IDs in TinyDB
    doc_ids = [doc.id for doc in docs]
    query_log.insert({'query': user_query, 'doc_ids': doc_ids})

    # Extract context from the documents
    context_list = [doc.page_content for doc in docs]
    print("\n" + "="*50)
    print(f"Context List : {context_list}")
    print("="*50 + "\n")
    context = "\n".join(context_list)
    print("\n" + "="*50)
    print(f"Context : {context}")
    print("="*50 + "\n")

    # Construct the prompt
    qa_prompt = PROMPT.format(context=context, question=user_query)
    print("\n" + "="*50)
    print(f"qa_prompt : {qa_prompt}")
    print("="*50 + "\n")

    response = together_ai.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": qa_prompt}]
    )
    print("\n" + "="*50)
    print(f"response : {response}")
    print("="*50 + "\n")
    bot_response = response.choices[0].message.content
    print("\n" + "="*50)
    print(f"bot_response : {bot_response}")
    print("="*50 + "\n")

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

# Example function to retrieve document IDs for a given query
def get_document_ids(query):
    QueryObj = Query()
    result = query_log.search(QueryObj.query == query)
    return result[0]['doc_ids'] if result else None

# Display document IDs for the latest query
if user_query:
    st.write("Document IDs for the latest query:")
    st.write(get_document_ids(user_query))


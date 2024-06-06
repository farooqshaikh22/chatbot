from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework

st.title("Langchain Demo with Gemma")
input_text = st.text_input("Search the topic you want")

## Ollama Gemma LLM

llm = Ollama(model = "Gemma:7B")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
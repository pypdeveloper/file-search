import streamlit as st
import os
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Upload a document and search it using AI")


input = st.text_input("Enter Search prompt")


if input:
    loader = PyPDFLoader("./data/file.pdf")
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    search = faiss_index.similarity_search(input)
    x = 0
    while x <= len(search):
        if x == len(search):
            break
        with st.container():
            st.write(search[x].page_content)
            st.write(search[x].metadata)
            st.divider()
        x+=1

template = """
Based on the information: {information}
Answer the question: {question}
"""

llm = OpenAI(temperature=0.8, verbose=True)

prompt = PromptTemplate(template=template, input_variables=['information', 'question'])
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if input: 
    response = chain.run({"information": search, "question": input})
    st.write(response)
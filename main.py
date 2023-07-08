import streamlit as st
import os
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Upload a document and search it using AI")

question = st.text_input("Enter Search prompt")

db_array = []

if st.button("Index files"):
    # for document in os.listdir("data"):
    #     path = os.path.join("data", document) 
    #     if os.path.isfile(path):
    #         if path == "data/.DS_Store":
    #             pass
    #         else: 
    #             global loader
    #             global pages
    #             global faiss_index

                # loader = PyPDFLoader(os.path.join("./data", document))
    loader = DirectoryLoader("./data", loader_cls=PyPDFLoader)
    pages = loader.load_and_split()
    st.session_state.faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    st.success("Docments Indexed")

if question:
    # for document in os.listdir("data"):
    #     st.write(document)
    #     path = os.path.join("data", document) 
    #     if os.path.isfile(path):
    #         if path == "data/.DS_Store":
    #             pass
    #         else:
    #             loader = PyPDFLoader(os.path.join("./data", document))
    #             pages = loader.load_and_split()
    #             faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
     
    search = st.session_state.faiss_index.similarity_search(question)
    
    x = 0
    information = ""
    while x <= len(search):
        if x == len(search):
            break
        information += search[x].page_content
        with st.container():
            st.write(search[x].page_content)
            st.write(search[x].metadata)
            st.divider()
        x+=1

template = """
Based on the information: {information}
Answer the question: {question}
"""

memory = ConversationBufferMemory()

llm = OpenAI(temperature=0.8, verbose=True)

prompt = PromptTemplate(template=template, input_variables=["information", "question"])
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if question:
    response = chain.run({"information": information, "question": question})
    st.write(response)
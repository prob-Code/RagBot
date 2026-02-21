import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Upload your C++ document and ask questions about it")

@st.cache_resource
def load_vectorstore():
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    final_documents = text_splitter.split_documents(documents=data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(final_documents, embeddings)
    return db

db = load_vectorstore()


llm = Ollama(model="gemma2:2b")

 
user_question =st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking..."):
        docs = db.similarity_search(user_question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context provided.
        context: {context}
        question: {user_question}
        answer:"""
        answer = llm.invoke(prompt)
        st.write(answer)
        

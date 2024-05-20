import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("Stock Reasearch Tool 💹")

st.sidebar.title("Stock News URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
processed_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)
if processed_url_clicked:
    loaders = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading....Started...✅✅✅")
    data = loaders.load()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitting....Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector....Started...✅✅✅")
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    vectorindex_openai.save_local("vectorstore")
    
    

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("vectorstore"):
        x = FAISS.load_local("vectorstore", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        retriever = x.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=retriever)
        result = chain({"question":query}, return_only_outputs=True)
        # {"answer":"","sources":[]}
        st.header("Answer: ")
        st.write(result["answer"])
        sources = result.get("sources","")
        if sources:
            st.subheader("Sources: ")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
        
        
        
        
        
        
        
        
        
        
    






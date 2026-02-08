import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import tempfile
import os

# ------------------------
# Load Environment
# ------------------------
load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title(" RAG Chatbot ")

# ------------------------
# Initialize LLM
# ------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ------------------------
# Load Embedding Model
# ------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(documents)

        # Create Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever()

        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        st.success("PDF processed successfully!")

        # ------------------------
        # Chat Interface
        # ------------------------
        query = st.text_input("Ask a question from the PDF")

        if query:
            with st.spinner("Thinking..."):
                result = qa_chain.run(query)
                st.write("### Answer:")
                st.write(result)

        os.remove(pdf_path)

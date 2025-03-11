import streamlit as st
import pdfplumber
import docx
import os
import faiss
import numpy as np
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
from langchain.docstore.in_memory import InMemoryDocstore

# ---- CONFIGURATION ----
GROQ_API_KEY = "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"
llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", groq_api_key=GROQ_API_KEY)

# Load SBERT Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS Database Path
DB_PATH = "qna_db"

# ---- FUNCTIONS ----
def load_faiss_db():
    """Load or initialize FAISS database."""
    if os.path.exists(f"{DB_PATH}/index.faiss"):
        return FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        embedding_size = len(embedding_model.embed_query("test"))  # Get embedding dimension
        index = faiss.IndexFlatL2(embedding_size)
        docstore = InMemoryDocstore()  # Fixed docstore issue
        return FAISS(embedding_model, index, docstore, {})

vector_db = load_faiss_db()

def extract_text(file):
    """Extract text from PDF, TXT, or DOCX."""
    text = ""
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            extracted_pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
            text = "\n".join(extracted_pages)
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

def process_document(text):
    """Split document into chunks and store in FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    vector_db.add_texts(chunks)
    vector_db.save_local(DB_PATH)

def retrieve_relevant_chunks(query):
    """Retrieve relevant document chunks from FAISS."""
    results = vector_db.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

def generate_answer(question, context):
    """Use Groq LLM to generate an answer using retrieved context."""
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    response = llm([SystemMessage(content="Answer the question based on provided context."), HumanMessage(content=prompt)])
    return response.content

# ---- STREAMLIT UI ----
st.title("📖 RAG-Based QnA Bot with Groq")

# Upload document
uploaded_file = st.file_uploader("Upload a document (PDF, TXT, or DOCX)", type=["pdf", "txt", "docx"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    text = extract_text(uploaded_file)
    if text:
        process_document(text)
        st.success("✅ Document processed and stored in FAISS!")

# Question Input
question = st.text_input("Ask a question related to the uploaded document:")

if st.button("Get Answer"):
    if question:
        st.subheader("🔍 Searching for relevant information...")
        context_chunks = retrieve_relevant_chunks(question)
        if context_chunks:
            full_context = " ".join(context_chunks)
            st.subheader("📝 Generated Answer:")
            answer = generate_answer(question, full_context)
            st.write(answer)
        else:
            st.warning("No relevant information found.")
    else:
        st.warning("Please enter a question.")

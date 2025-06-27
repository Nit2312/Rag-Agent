import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ------------------ Initialization ------------------ #
st.set_page_config(page_title="Agentic RAG Chat", layout="wide")
st.title("📄 Agentic RAG - PDF Q&A Chat")
st.secrets["GROQ_API_KEY"]

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ------------------ Session Setup ------------------ #
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chain", None)

# ------------------ Cached Resources ------------------ #
@st.cache_data(show_spinner="⏳ Loading and parsing PDFs...")
def process_uploaded_pdfs(uploaded_files):
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            all_docs.extend(loader.load())
    return all_docs

@st.cache_resource(show_spinner="🔍 Loading HuggingFace Embeddings...")
def get_embeddings_model():
    return HuggingFaceEmbeddings()

@st.cache_resource(show_spinner="🔎 Creating vector store and chain...")
def create_conversational_chain(_docs, _embeddings_model):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_docs)
    vectorstore = FAISS.from_documents(chunks, _embeddings_model)

    llm = ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=GROQ_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ------------------ Chat Functionality ------------------ #
def handle_user_input():
    user_input = st.chat_input("Ask a question about the uploaded PDFs...")
    if user_input:
        result = st.session_state.chain.invoke({"question": user_input})
        answer = result.get("answer", "").strip()

        if not answer or "i don't know" in answer.lower() or len(answer) < 5:
            answer = "❗ Sorry, I couldn't find an answer to that in the uploaded documents."

        st.session_state.chat_history.extend([("You", user_input), ("AI", answer)])

def display_chat_history():
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(message)

# ------------------ Export Utilities ------------------ #
def export_chat_to_pdf(chat_history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    text = c.beginText(40, 800)
    text.setFont("Helvetica", 12)
    for sender, message in chat_history:
        for line in f"{sender}: {message}".splitlines():
            text.textLine(line)
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def export_chat_to_excel(chat_history):
    df = pd.DataFrame(chat_history, columns=["Sender", "Message"])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ChatHistory")
    buffer.seek(0)
    return buffer

# ------------------ Main App Logic ------------------ #
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and "docs" not in st.session_state:
    st.session_state.docs = process_uploaded_pdfs(uploaded_files)
    st.session_state.embeddings_model = get_embeddings_model()
    st.session_state.chain = create_conversational_chain(
        st.session_state.docs,
        st.session_state.embeddings_model
    )
    st.success("✅ PDFs processed and ready for Q&A!")

if st.session_state.chain:
    handle_user_input()
    display_chat_history()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📤 Export as PDF"):
            pdf = export_chat_to_pdf(st.session_state.chat_history)
            st.download_button("Download PDF", pdf, file_name="chat_session.pdf", mime="application/pdf")

    with col2:
        if st.button("📥 Export as Excel"):
            excel = export_chat_to_excel(st.session_state.chat_history)
            st.download_button("Download Excel", excel, file_name="chat_session.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.rerun()

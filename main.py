import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# ------------------ LangChain Imports (modern) ------------------ #
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Conversational RAG chain (modern modular import)
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.memory import ConversationBufferMemory

# ------------------ Streamlit Setup ------------------ #
st.set_page_config(page_title="Agentic RAG Chat", layout="wide")
st.title("📄 Agentic RAG - PDF Q&A Chat")

# ------------------ API Key Load ------------------ #
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not GROQ_API_KEY:
    st.error("🚫 Groq API key not found in Streamlit secrets or environment variables.")
    st.stop()

# ------------------ Session Setup ------------------ #
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chain", None)
st.session_state.setdefault("docs", None)
st.session_state.setdefault("embeddings_model", None)

# ------------------ Cached Resources ------------------ #
@st.cache_data(show_spinner="⏳ Parsing uploaded PDFs...")
def process_uploaded_pdfs(uploaded_files):
    """Load and parse multiple PDFs using PyPDFLoader."""
    all_docs = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp.flush()
                loader = PyPDFLoader(tmp.name)
                all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {str(e)}")
    return all_docs


@st.cache_resource(show_spinner="🔍 Loading HuggingFace Embeddings...")
def get_embeddings_model():
    try:
        # all-MiniLM-L6-v2 is lightweight and works great
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {str(e)}")
        return None


@st.cache_resource(show_spinner="🧠 Building RAG Chain...")
def create_conversational_chain(_docs, _embeddings_model):
    """Create FAISS vectorstore and conversational chain."""
    if not _docs:
        st.error("No documents loaded.")
        return None

    if not _embeddings_model:
        st.error("Embeddings model not available.")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(_docs)
        vectorstore = FAISS.from_documents(chunks, _embeddings_model)

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="Llama3-8b-8192",
            temperature=0
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
        )

        return chain

    except Exception as e:
        st.error(f"❌ Failed to create conversation chain: {str(e)}")
        return None


# ------------------ Chat Functionality ------------------ #
def handle_user_input():
    user_input = st.chat_input("Ask a question about the uploaded PDFs...")
    if not user_input or not st.session_state.chain:
        return

    st.session_state.chat_history.append(("You", user_input))

    try:
        result = st.session_state.chain.invoke({"question": user_input})

        # Handle dict or string outputs
        if isinstance(result, dict):
            answer = (
                result.get("answer")
                or result.get("result")
                or result.get("output_text")
                or result.get("text", "")
            )
        else:
            answer = str(result)

        if not answer or len(answer.strip()) < 5:
            answer = "❗ Sorry, I couldn't find an answer in the uploaded documents."

        st.session_state.chat_history.append(("AI", answer.strip()))
    except Exception as e:
        st.session_state.chat_history.append(("AI", f"❌ Error: {str(e)}"))


def display_chat_history():
    for sender, message in st.session_state.chat_history:
        role = "user" if sender.lower() in ("you", "user") else "assistant"
        with st.chat_message(role):
            st.markdown(message)


# ------------------ Export Utilities ------------------ #
def export_chat_to_pdf(chat_history):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        text = c.beginText(40, 800)
        text.setFont("Helvetica", 11)

        for sender, message in chat_history:
            for line in f"{sender}: {message}".splitlines():
                text.textLine(line[:120])  # wrap to avoid overflow
        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"❌ PDF export failed: {str(e)}")
        return None


def export_chat_to_excel(chat_history):
    try:
        df = pd.DataFrame(chat_history, columns=["Sender", "Message"])
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="ChatHistory")
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"❌ Excel export failed: {str(e)}")
        return None


# ------------------ Main App Logic ------------------ #
uploaded_files = st.file_uploader("📎 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.docs = process_uploaded_pdfs(uploaded_files)
    st.session_state.embeddings_model = get_embeddings_model()
    st.session_state.chain = create_conversational_chain(
        st.session_state.docs,
        st.session_state.embeddings_model
    )
    if st.session_state.chain:
        st.success("✅ PDFs processed and RAG chain created successfully!")

if st.session_state.chain:
    handle_user_input()
    display_chat_history()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📤 Export as PDF"):
            pdf = export_chat_to_pdf(st.session_state.chat_history)
            if pdf:
                st.download_button(
                    "Download PDF",
                    pdf,
                    file_name="chat_session.pdf",
                    mime="application/pdf"
                )

    with col2:
        if st.button("📥 Export as Excel"):
            excel = export_chat_to_excel(st.session_state.chat_history)
            if excel:
                st.download_button(
                    "Download Excel",
                    excel,
                    file_name="chat_session.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.rerun()
else:
    st.info("👆 Upload PDF files above to begin.")

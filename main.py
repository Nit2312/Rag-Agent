import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# -------- LangChain & related imports -------- #
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Import retrieval-chain components
from langchain.chains.retrieval import create_retrieval_chain  # correct path for newer LangChain :contentReference[oaicite:1]{index=1}
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware import create_history_aware_retriever  # hypothetical path; adjust if your version differs
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------- Initialization -------- #
st.set_page_config(page_title="Agentic RAG Chat", layout="wide")
st.title("📄 Agentic RAG – PDF Q&A Chat")

load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
if not GROQ_API_KEY:
    st.error("🚫 Groq API key not found. Please check secrets or env.")
    st.stop()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chain", None)
st.session_state.setdefault("docs", None)
st.session_state.setdefault("embeddings_model", None)

# -------- Cached Resource Functions -------- #
@st.cache_data(show_spinner="⏳ Loading and parsing PDFs...")
def process_uploaded_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp.flush()
                loader = PyPDFLoader(tmp.name)
                docs.extend(loader.load())
        except Exception as e:
            st.error(f"❌ Error loading PDF {file.name}: {e}")
    return docs

@st.cache_resource(show_spinner="🔍 Loading Embeddings Model...")
def get_embeddings_model():
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {e}")
        return None

@st.cache_resource(show_spinner="🔎 Creating vector store + chain...")
def create_conversational_chain(_docs, _embeddings_model):
    if not _docs or not _embeddings_model:
        st.error("Documents or embeddings model missing.")
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(_docs)
        vectorstore = FAISS.from_documents(chunks, _embeddings_model)

        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192", temperature=0)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 1. Build question reformulation prompt
        contextualize_q_system = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood without the chat history. "
            "DO NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # 2. Build answer generation prompt
        answer_system = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don’t know the answer, say that you don’t know. Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", answer_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        document_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)

        # 3. Combine into the RAG chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=document_chain
        )

        return rag_chain

    except Exception as e:
        st.error(f"❌ Failed to create conversation chain: {e}")
        return None

# -------- Chat Functionality -------- #
def handle_user_input():
    user_input = st.chat_input("Ask a question about the uploaded PDFs…")
    if user_input and st.session_state.chain:
        try:
            result = st.session_state.chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            # Extract answer string
            answer = None
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("output_text") or result.get("result")
            else:
                answer = str(result)
            if not answer or len(answer.strip()) < 5 or "i don't know" in answer.lower():
                answer = "❗ Sorry, I couldn’t find an answer in the uploaded documents."

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", answer.strip()))
        except Exception as e:
            st.session_state.chat_history.append(("AI", f"❌ Error: {e}"))

def display_chat_history():
    for sender, message in st.session_state.chat_history:
        role = "user" if sender.lower() in ("you", "user") else "assistant"
        with st.chat_message(role):
            st.markdown(message)

# -------- Export Utilities -------- #
def export_chat_to_pdf(chat_history):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        text = c.beginText(40, 800)
        text.setFont("Helvetica", 11)
        for sender, message in chat_history:
            for line in f"{sender}: {message}".splitlines():
                text.textLine(line[:120])
        c.drawText(text)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"❌ Export to PDF failed: {e}")
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
        st.error(f"❌ Export to Excel failed: {e}")
        return None

# -------- Main App Logic -------- #
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.session_state.chain is None:
    st.session_state.docs = process_uploaded_pdfs(uploaded_files)
    st.session_state.embeddings_model = get_embeddings_model()
    st.session_state.chain = create_conversational_chain(
        st.session_state.docs,
        st.session_state.embeddings_model
    )
    if st.session_state.chain:
        st.success("✅ PDFs processed and ready for Q&A!")

if st.session_state.chain:
    handle_user_input()
    display_chat_history()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📤 Export as PDF"):
            pdf_buf = export_chat_to_pdf(st.session_state.chat_history)
            if pdf_buf:
                st.download_button("Download PDF", pdf_buf, file_name="chat_session.pdf", mime="application/pdf")

    with col2:
        if st.button("📥 Export as Excel"):
            excel_buf = export_chat_to_excel(st.session_state.chat_history)
            if excel_buf:
                st.download_button("Download Excel", excel_buf, file_name="chat_session.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.rerun()
else:
    st.info("👆 Upload PDF files above to begin.")

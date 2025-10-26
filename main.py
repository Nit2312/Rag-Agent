import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# LangChain and related imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Latest LangChain imports for RAG
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------ Initialization ------------------ #
st.set_page_config(page_title="Agentic RAG Chat", layout="wide")
st.title("📄 Agentic RAG - PDF Q&A Chat")

# Securely load API key
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not GROQ_API_KEY:
    st.error("🚫 Groq API key not found in secrets or environment. Please check your configuration.")
    st.stop()

# ------------------ Session Setup ------------------ #
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chain", None)

# ------------------ Cached Resources ------------------ #
@st.cache_data(show_spinner="⏳ Loading and parsing PDFs...")
def process_uploaded_pdfs(uploaded_files):
    all_docs = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                loader = PyPDFLoader(tmp.name)
                all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"❌ Error loading PDF: {file.name} — {str(e)}")
    return all_docs


@st.cache_resource(show_spinner="🔍 Loading HuggingFace Embeddings...")
def get_embeddings_model():
    try:
        return HuggingFaceEmbeddings()
    except Exception as e:
        st.error(f"❌ Failed to load embeddings: {str(e)}")
        st.stop()


@st.cache_resource(show_spinner="🔎 Creating vector store and chain...")
def create_conversational_chain(_docs, _embeddings_model):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(_docs)
        vectorstore = FAISS.from_documents(chunks, _embeddings_model)

        llm = ChatGroq(
            temperature=0,
            model_name="Llama3-8b-8192",
            api_key=GROQ_API_KEY
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 1️⃣ Create contextual question reformulation chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. DO NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt,
        )

        # 2️⃣ Create document combination (answer generation) chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        document_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

        # 3️⃣ Combine both into a retrieval-based chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_documents_chain=document_chain,
        )

        return rag_chain

    except Exception as e:
        st.error(f"❌ Failed to create the conversation chain: {str(e)}")
        st.stop()


# ------------------ Chat Functionality ------------------ #
def handle_user_input():
    user_input = st.chat_input("Ask a question about the uploaded PDFs...")
    if user_input and st.session_state.chain:
        try:
            result = st.session_state.chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

            answer = result.get("answer", "").strip() or result.get("output_text", "").strip()

            if not answer or "i don't know" in answer.lower() or len(answer) < 5:
                answer = "❗ Sorry, I couldn't find an answer to that in the uploaded documents."

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", answer))
        except Exception as e:
            st.session_state.chat_history.append(("AI", f"❌ Error: {str(e)}"))


def display_chat_history():
    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(message)


# ------------------ Export Utilities ------------------ #
def export_chat_to_pdf(chat_history):
    try:
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
    except Exception as e:
        st.error(f"❌ Failed to export PDF: {str(e)}")
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
        st.error(f"❌ Failed to export Excel: {str(e)}")
        return None


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
            if pdf:
                st.download_button("Download PDF", pdf, file_name="chat_session.pdf", mime="application/pdf")

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

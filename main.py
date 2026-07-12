import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Document RAG System", page_icon="📚", layout="wide")
st.title("📚 Custom Document Question Answering System (RAG)")
st.write(
    "Upload your private PDFs (Notes, Resumes, Papers) and ask questions grounded in your data."
)

# Sidebar for Document Ingestion
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            with st.spinner("Processing document..."):
                # Prepare file payload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                        st.info(f"Chunks indexed: {response.json()['chunks_created']}")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend API. Is FastAPI running?")

# Main Chat Interface
st.subheader("Chat with your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_question := st.chat_input("Ask something about your documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Fetch response from FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query", json={"question": user_question}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    
                    # Optional: Expand to see source context
                    with st.expander("View Source Context Links"):
                        for doc in response.json()["source_documents"]:
                            st.caption(f"**Page {doc['page'] + 1}:** {doc['text']}...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Error generating answer from backend.")
            except requests.exceptions.ConnectionError:
                st.error("Backend API is unreachable.")
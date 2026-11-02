import streamlit as st
import requests

st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="📚",
    layout="wide"
)

API_BASE = "http://127.0.0.1:8000"

st.sidebar.title("📁 Document Manager")

# File Upload Section
uploaded_file = st.sidebar.file_uploader("Upload New Research PDF", type=["pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Ingest PDF"):
        with st.spinner("Uploading and updating index..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                if res.status_code == 200:
                    st.sidebar.success(f"Successfully uploaded {uploaded_file.name}!")
                    st.rerun()
                else:
                    st.sidebar.error(res.json().get("detail", "Upload failed."))
            except Exception as e:
                st.sidebar.error(f"Connection error: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Ingested Papers")

# Fetch and display current files
try:
    files_res = requests.get(f"{API_BASE}/files", timeout=5)
    if files_res.status_code == 200:
        file_list = files_res.json()
        if file_list:
            for fname in file_list:
                col1, col2 = st.sidebar.columns([4, 1])
                col1.text(fname)
                if col2.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                    try:
                        del_res = requests.delete(f"{API_BASE}/files/{fname}", timeout=10)
                        if del_res.status_code == 200:
                            st.sidebar.success(f"Deleted {fname}")
                            st.rerun()
                        else:
                            st.sidebar.error("Failed to delete file.")
                    except Exception as e:
                        st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.info("No documents found in raw directory.")
    else:
        st.sidebar.warning("Could not fetch file list from API.")
except Exception:
    st.sidebar.error("Backend server is offline.")

# Main Chat Interface
st.title("📚 Research Paper RAG Assistant")
st.markdown("Ask questions across your active research papers library and review verified citations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know from your research papers?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and synthesizing answer..."):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={"query": prompt, "similarity_top_k": 3},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer_output = data.get("response_with_citations", "No response returned.")
                    st.markdown(answer_output)
                    st.session_state.messages.append({"role": "assistant", "content": answer_output})
                else:
                    error_msg = f"API Error ({response.status_code}): {response.json().get('detail', response.text)}"
                    st.error(error_msg)
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to FastAPI backend. Ensure uvicorn is running on port 8000.")
            except requests.exceptions.Timeout:
                st.error("The request timed out while waiting for local LLM generation.")
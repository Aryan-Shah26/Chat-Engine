import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat Engine", layout="wide")
st.title("RAG Chat Engine")


@st.cache_data(ttl=5)
def fetch_collections():
    return requests.get(f"{API_URL}/collections").json()


with st.sidebar:
    st.header("Collections")
    collections = fetch_collections()
    names = {c["name"]: c["id"] for c in collections}

    new_name = st.text_input("New collection name")
    if st.button("Create collection") and new_name:
        requests.post(f"{API_URL}/collections", json={"name": new_name})
        st.cache_data.clear()
        st.rerun()

    if not names:
        st.info("Create a collection to get started.")
        st.stop()

    selected_name = st.selectbox("Active collection", list(names.keys()))
    collection_id = names[selected_name]

    st.divider()
    uploaded = st.file_uploader("Upload documents", type=["pdf", "html", "txt", "md"], accept_multiple_files=True)
    if uploaded and st.button("Ingest"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}..."):
                resp = requests.post(
                    f"{API_URL}/collections/{collection_id}/documents",
                    files={"file": (f.name, f.getvalue())},
                )
            if resp.ok:
                st.success(f"Indexed {f.name}: {resp.json()['chunk_count']} chunks")
            else:
                st.error(resp.json().get("detail", "Ingestion failed"))

    st.divider()
    mode = st.radio("Query mode", ["hybrid", "multi_query", "multi_source", "agentic"])
    verify = st.checkbox("Verify citations", value=True)

question = st.text_input("Ask a question:")
if question:
    with st.spinner("Thinking..."):
        resp = requests.post(
            f"{API_URL}/collections/{collection_id}/chat",
            json={"question": question, "mode": mode, "verify_citations": verify},
        )
    if resp.ok:
        data = resp.json()
        st.subheader("Answer")
        st.write(data["answer"])
        st.subheader("Sources")
        for s in data["sources"]:
            st.caption(s)
    else:
        st.error(resp.json().get("detail", "Query failed"))

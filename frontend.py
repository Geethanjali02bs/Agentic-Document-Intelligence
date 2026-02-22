import streamlit as st
import requests

st.title("📄 Document AI Agent")

# ⭐ Upload PDF UI
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    files = {"file": uploaded_file}
    res = requests.post("http://127.0.0.1:8000/upload", files=files)

    if res.status_code == 200:
        st.success(res.json()["message"])
    else:
        st.error("Upload failed")

# ⭐ Query UI
query = st.text_input("Ask a question from document")

if query:
    res = requests.post(
        "http://127.0.0.1:8000/query",
        params={"query": query}
    )

    if res.status_code == 200:
        data = res.json()

        st.subheader("🤖 Answer")
        st.write(data["answer"])

        st.subheader("📌 Sources")
        for s in data["sources"]:
            st.write(f"Page {s['page_label']} — {s['file_name']}")
    else:
        st.error("Query failed")
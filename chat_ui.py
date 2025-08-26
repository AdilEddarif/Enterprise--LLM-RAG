# chat_ui.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 PDF RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Pose ta question...")

if query:
    with st.spinner("Recherche..."):
        try:
            res = requests.post(API_URL, json={"query": query}, timeout=60)
            res.raise_for_status()
            data = res.json()
            answer = data.get("answer", "Erreur.")
            source = data.get("source", "")
        except Exception as e:
            answer = f"[Erreur API] {e}"
            source = ""

    st.session_state.history.append({"q": query, "a": answer, "s": source})

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        st.write(h["a"])
        if h["s"]:
            st.caption(f"SOURCE: {h['s']}")

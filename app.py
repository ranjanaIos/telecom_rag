import streamlit as st
from rag_pipeline import load_docs, split_docs, create_vector_db, create_chain

st.set_page_config(page_title="Telecom GenAI Assistant")

st.title("📡 Telecom Customer Retention – GenAI Assistant")

if st.button("Create Knowledge Base"):
    docs = load_docs("data/telecom.pdf")
    chunks = split_docs(docs)
    create_vector_db(chunks)
    st.success("Knowledge Base Created Successfully!")

if "qa" not in st.session_state:
    st.session_state.qa = create_chain()

question = st.text_input("Ask a question about telecom customer retention")

if question:
    with st.spinner("Generating answer..."):
        answer = st.session_state.qa(question)
        st.write(answer)

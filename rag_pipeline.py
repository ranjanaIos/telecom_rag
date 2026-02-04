from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------- SAFE EMBEDDINGS (NO TORCH) ----------------
class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t))] * 8 for t in texts]

    def embed_query(self, text):
        return [float(len(text))] * 8


PERSIST_DIR = "chroma_db"

import os

def load_docs(path):
    abs_path = os.path.abspath(path)
    loader = PyPDFLoader(abs_path)
    return loader.load()



def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_db(docs):
    embeddings = SimpleEmbeddings()

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb


def create_chain():
    embeddings = SimpleEmbeddings()

    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    prompt = PromptTemplate(
        template="""
You are a Telecom Industry Research Assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer with:
- Summary
- Key Factors
- Practical Recommendation
""",
        input_variables=["context", "question"]
    )

    def qa_fn(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        response = llm.invoke(final_prompt)
        return response.content

    return qa_fn

# 📡 Telecom Customer Retention – GenAI RAG Chatbot
## 📌 Project Overview

The Indian telecom industry faces intense competition, high customer churn, and declining revenues. Understanding customer retention is critical for telecom operators to sustain profitability and long-term growth.

This project implements a Generative AI–powered chatbot using LangChain and Retrieval-Augmented Generation (RAG) to provide context-aware insights from academic research on customer retention in the Indian telecom sector.

The chatbot assists:

- Academic researchers
- Telecom managers
- Industry practitioners

by answering domain-specific questions grounded strictly in telecom research literature.

## 🎯 Problem Statement

The Indian telecom industry serves over 1.15 billion users, yet faces challenges such as:

- High customer churn
- Pricing pressure
- Network quality issues
- Regulatory constraints
- Changing customer expectations

A comprehensive research study identified 64 sub-factors, grouped into 7 major domains, that influence customer retention.

### Objective:
Develop a GenAI-based chatbot using LangChain and RAG that can retrieve, analyze, and generate actionable insights from telecom research documents.

## 🧠 Key Features

- 📄 Loads telecom research PDF as knowledge source
- 🔍 Retrieves relevant context using vector similarity search
- 🤖 Generates accurate answers using Llama 3.1 (Groq API)
- 🧠 Prevents hallucination by grounding responses in documents
- 💬 Interactive chat interface using Streamlit

## 🏗️ System Architecture
      User Question
            ↓
      Streamlit UI
            ↓
      Retriever (Chroma Vector DB)
            ↓
      Relevant Research Chunks
            ↓
      Prompt + Context Injection
            ↓
      Llama 3.1 (Groq API)
            ↓
      Grounded Answer

## 🛠️ Tech Stack

| Component            | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| GenAI Model          | Llama 3.1 (Groq API – Free)   |
| Framework            | LangChain                     |
| Vector Database      | Chroma (local, persistent)    |
| Embeddings           | Lightweight custom embeddings |
| UI                   | Streamlit                     |
| Document Loader      | PyPDF                         |

## ⚙️ How It Works

1. The research document is loaded and split into chunks
2. Chunks are stored in a vector database
3. User query retrieves the most relevant chunks
4. Retrieved context is injected into the LLM prompt
5. LLM generates a grounded, domain-specific response

## ▶️ How to Run the Project

### Install Dependencies
pip install -r requirements.txt

### 2️⃣ Set Groq API Key (Secure Method)
#### macOS / Linux
export GROQ_API_KEY="gsk_your_api_key_here"

### 3️⃣ Add Telecom Research PDF
Place the research file inside:
data/telecom.pdf

### 4️⃣ Run Application
streamlit run app.py

### 5️⃣ Initialize Knowledge Base
Click “Create Knowledge Base” once in the UI.

## 💡 Sample Questions to Ask

1. What are the main factors affecting customer retention in Indian telecom?
2. How does pricing influence customer churn?
3. What role does CRM play in customer loyalty?
4. How did Reliance Jio impact customer retention strategies?
5. Recommend retention strategies for telecom operators.

## 🧾 Conclusion

This project successfully demonstrates how Generative AI combined with Retrieval-Augmented Generation can be applied to extract actionable insights from academic research. The chatbot offers a scalable and intelligent solution for understanding customer retention dynamics in the Indian telecom industry.

## 👩‍💻 Author

### Ranjana Patidar
Data Science & Generative AI Practitioner

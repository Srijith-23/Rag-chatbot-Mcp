# Agentic RAG Chatbot for Multi-Format Document QA using Model Context Protocol (MCP)

## 📌 Problem Statement

Build an **agent-based Retrieval-Augmented Generation (RAG) chatbot** that answers user questions using uploaded documents in various formats. The architecture must follow an **agentic design** with **Model Context Protocol (MCP)** for structured message passing between agents and/or agents ↔ LLMs.

---

## ✅ Core Functional Requirements

- **Support Uploading & Parsing of Diverse Document Formats:**
  - PDF
  - PPTX
  - CSV
  - DOCX
  - TXT / Markdown

- **Agentic Architecture (minimum 3 agents):**
  - **IngestionAgent**: Parses and preprocesses documents.
  - **RetrievalAgent**: Handles embeddings + semantic retrieval from vector store.
  - **LLMResponseAgent**: Forms final LLM query using retrieved context and generates answer.

- **Use Model Context Protocol (MCP):**
  - Agents communicate using structured context objects:
    ```json
    {
      "sender": "RetrievalAgent",
      "receiver": "LLMResponseAgent",
      "type": "CONTEXT_RESPONSE",
      "trace_id": "abc-123",
      "payload": {
        "top_chunks": ["...", "..."],
        "query": "What are the KPIs?"
      }
    }
    ```
  - MCP can be implemented via in-memory messaging, REST endpoints, or pub/sub systems.

- **Vector Store + Embeddings**
  - Use any embeddings model (OpenAI, HuggingFace, etc.)
  - Store and query embeddings in FAISS, Chroma, or other vector DBs.

- **Chatbot Interface (UI)**
  - Upload documents
  - Ask multi-turn questions
  - View responses with source context
  - UI Framework options: Flask, Streamlit, React, etc.

---

## ⚙️ Tech Stack (Example)

- Python 3.10+
- Flask (or Streamlit) for UI
- LangChain / FAISS / HuggingFace
- OpenAI or local embeddings
- PowerShell / Bash for setup

---

## 🗂️ Project Structure (Example)


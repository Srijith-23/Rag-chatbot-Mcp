# ------------------ Imports ------------------
import os
import uuid
from flask import Flask, request, render_template_string, session
from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, CSVLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from transformers import pipeline

# ------------------ Configuration ------------------
UPLOAD_FOLDER = "uploads"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ MCP Message Class ------------------
class MCPMessage:
    def __init__(self, sender, receiver, type_, trace_id, payload):
        self.sender = sender
        self.receiver = receiver
        self.type = type_
        self.trace_id = trace_id
        self.payload = payload

# ------------------ Ingestion Agent ------------------
class IngestionAgent:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def parse(self, filepaths):
        docs = []
        for path in filepaths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".txt":
                loader = TextLoader(path)
            elif ext == ".pdf":
                loader = PyMuPDFLoader(path)
            elif ext == ".csv":
                loader = CSVLoader(path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(path)
            elif ext == ".pptx":
                loader = UnstructuredPowerPointLoader(path)
            else:
                continue
            raw_docs = loader.load()
            docs.extend(self.splitter.split_documents(raw_docs))
        return docs
# ------------------ Retrieval Agent ------------------
class RetrievalAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.db = None
        self.reranker = CrossEncoder(RERANK_MODEL)

    def build_vector_store(self, docs):
        print("✅ Building FAISS index...")
        self.db = FAISS.from_documents(docs, self.embeddings)
        print("✅ FAISS index built successfully.")

    def retrieve(self, query, top_k=5):
        if not self.db:
            raise ValueError("Vector store is empty. Upload and index documents first.")

        retriever = self.db.as_retriever()
        raw_docs = retriever.get_relevant_documents(query)
        print(f"✅ Retrieved {len(raw_docs)} raw chunks.")

        if not raw_docs:
            return []

        # Rerank with CrossEncoder
        pairs = [(query, doc.page_content) for doc in raw_docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [doc.page_content for doc, _ in ranked[:top_k]]
        print(f"✅ Top {top_k} chunks selected after reranking.")
        return top_chunks

# ------------------ LLM Response Agent ------------------
# ------------------ LLM Response Agent ------------------
class LLMResponseAgent:
    def __init__(self):
        print("✅ Loading local extractive QA pipeline...")
        self.qa_pipeline = pipeline("question-answering", model=QA_MODEL)
        print("✅ QA pipeline loaded.")

    def handle(self, message, chat_history):
        trace_id = message.trace_id
        query = message.payload["query"]
        top_chunks = message.payload["top_chunks"]

        if not top_chunks:
            answer = "Sorry, I couldn't find any relevant context to answer."
            return MCPMessage("LLMResponseAgent", "User", "FINAL_ANSWER", trace_id, {"answer": answer, "context": ""})

        # Combine retrieved chunks
        context = "\n\n".join(top_chunks)
        print(f"🧠 Using {len(top_chunks)} chunks, {len(context)} characters total.")

        # ✨ Improved Prompt
        better_prompt = f"""
You are an assistant doing Extractive QA.

Instructions:
- Answer ONLY from the CONTEXT.
- If asked about lists (e.g. achievements), find bullet points or enumerated text.
- Be complete but concise.
- If the answer is not in context, reply "I don't know".

CONTEXT:
{context}

QUESTION:
{query}
"""

        # Run QA
        result = self.qa_pipeline(question=query, context=context)
        answer = result.get("answer", "I don't know")

        return MCPMessage(
            sender="LLMResponseAgent",
            receiver="User",
            type_="FINAL_ANSWER",
            trace_id=trace_id,
            payload={"answer": answer.strip(), "context": context}
        )

# ------------------ Flask Templates ------------------
CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>Welcome to RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f6f6f6; margin: 0; }
        .header { background-color: #10a37f; color: white; padding: 20px; text-align: center; font-size: 24px; }
        .chat-container { max-width: 800px; margin: 20px auto; background: white; border-radius: 8px; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .chat-box { max-height: 400px; overflow-y: auto; margin-bottom: 20px; border: 1px solid #ccc; padding: 10px; background: #fafafa; border-radius: 5px; }
        .user-message { text-align: right; margin: 10px; }
        .assistant-message { text-align: left; margin: 10px; background-color: #e8f5e9; padding: 10px; border-radius: 5px; }
        form { display: flex; flex-direction: column; gap: 10px; }
        input[type="file"], input[type="text"], input[type="submit"] { padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
        input[type="submit"] { background-color: #10a37f; color: white; cursor: pointer; }
    </style>
</head>
<body>
    <div class="header">Welcome to RAG Chatbot</div>
    <div class="chat-container">
        <div class="chat-box">
            {% for entry in chat_history %}
                <div class="user-message"><strong>You:</strong> {{ entry['question'] }}</div>
                <div class="assistant-message"><strong>Assistant:</strong> {{ entry['answer'] }}</div>
            {% endfor %}
        </div>
        <form method="post" action="/interact" enctype="multipart/form-data">
            <input type="file" name="files" multiple>
            <input type="text" name="query" placeholder="Ask your question here..." required>
            <input type="submit" value="Submit">
        </form>
    </div>
</body>
</html>
"""

# ------------------ Flask Routes ------------------
@app.route("/", methods=["GET"])
def index():
    session["chat_history"] = []
    return render_template_string(CHAT_HTML, chat_history=[])

@app.route("/interact", methods=["POST"])
def interact():
    # Handle file upload
    files = request.files.getlist("files")
    filepaths = []
    for file in files:
        if file.filename:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)
            filepaths.append(path)

    # Ingest new docs if uploaded
    if filepaths:
        docs = ingestion_agent.parse(filepaths)
        retrieval_agent.build_vector_store(docs)
        session["chat_history"] = []

    # Handle question
    query = request.form.get("query")
    trace_id = str(uuid.uuid4())

    # Use existing DB
    try:
        top_chunks = retrieval_agent.retrieve(query)
    except Exception as e:
        top_chunks = []
        print(f"⚠️ Retrieval error: {e}")

    message = MCPMessage(
        sender="RetrievalAgent",
        receiver="LLMResponseAgent",
        type_="CONTEXT_RESPONSE",
        trace_id=trace_id,
        payload={"top_chunks": top_chunks, "query": query}
    )

    chat_history = session.get("chat_history", [])
    response = llm_response_agent.handle(message, chat_history)
    answer = response.payload["answer"]

    chat_history.append({"question": query, "answer": answer})
    session["chat_history"] = chat_history

    return render_template_string(CHAT_HTML, chat_history=chat_history)

# ------------------ Initialize Agents ------------------
ingestion_agent = IngestionAgent()
retrieval_agent = RetrievalAgent()
llm_response_agent = LLMResponseAgent()

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)


import os
import nest_asyncio
import chromadb
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate

nest_asyncio.apply()
load_dotenv()

app = FastAPI()

# ⭐ Models
Settings.llm = Groq(model="llama-3.3-70b-versatile")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

VECTOR_PATH = "./chroma_db"
FILE_PATH = "./data/uploaded.pdf"

# ⭐ Root endpoint
@app.get("/")
def home():
    return {"message": "RAG Backend Running"}

# ⭐ Upload endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # ⭐ SAFE reset vector DB (no folder delete)
    if os.path.exists(VECTOR_PATH):
        db = chromadb.PersistentClient(path=VECTOR_PATH)
        try:
            db.delete_collection("docs")
        except:
            pass

    # ⭐ Save uploaded file
    os.makedirs("./data", exist_ok=True)
    with open(FILE_PATH, "wb") as f:
        f.write(await file.read())

    # ⭐ Parse PDF
    parser = LlamaParse(result_type="markdown")
    documents = parser.load_data(FILE_PATH)

    # ⭐ Metadata for citations
    for i, doc in enumerate(documents):
        doc.metadata["file_name"] = file.filename
        doc.metadata["page_label"] = str(i + 1)

    # ⭐ Create vector DB
    db = chromadb.PersistentClient(path=VECTOR_PATH)
    collection = db.get_or_create_collection("docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return {"message": "PDF uploaded and indexed successfully"}

# ⭐ Query endpoint
@app.post("/query")
async def query_pdf(query: str):

    db = chromadb.PersistentClient(path=VECTOR_PATH)
    collection = db.get_or_create_collection("docs")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    index = VectorStoreIndex.from_vector_store(vector_store)

    # ⭐ Anti-hallucination prompt
    qa_prompt = PromptTemplate(
        "Answer ONLY using context.\n"
        "If not present say 'Not found'.\n\n"
        "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=7,
        text_qa_template=qa_prompt
    )

    response = query_engine.query(query)

    # ⭐ Remove duplicate citations
    sources = [node.metadata for node in response.source_nodes]
    unique_sources = {tuple(s.items()): s for s in sources}.values()

    return {
        "answer": str(response),
        "sources": list(unique_sources)
    }
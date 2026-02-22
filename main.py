import os
import nest_asyncio
import chromadb
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate

nest_asyncio.apply()
load_dotenv()

print("GROQ:", os.getenv("GROQ_API_KEY")[:10])
print("LLAMA:", os.getenv("LLAMA_CLOUD_API_KEY")[:10])

# ⭐ Setup models
Settings.llm = Groq(model="llama-3.3-70b-versatile")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ⭐ File path
file_path = "./data/report1.pdf"

if not os.path.exists(file_path):
    raise Exception("❌ report1.pdf not found inside data folder")

# ⭐ Chroma DB
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("docs")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ⭐ Parse PDF with LlamaParse
print("\n📄 Parsing PDF with LlamaParse...")
parser = LlamaParse(result_type="markdown")
documents = parser.load_data(file_path)

print(f"✅ Parsed {len(documents)} pages")

if len(documents) == 0:
    raise Exception("❌ No pages parsed")

# ⭐ FIXED metadata injection (correct page numbers)
for i, doc in enumerate(documents):
    doc.metadata["file_name"] = os.path.basename(file_path)
    doc.metadata["page_label"] = str(i + 1)

print("\n--- MARKDOWN SAMPLE ---")
print(documents[0].text[:200])

# ⭐ Build index with better chunking
print("\n🧠 Building index...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    chunk_size=512,
    chunk_overlap=50
)

# ⭐ Anti-hallucination prompt
qa_prompt = PromptTemplate(
    "Answer ONLY using the context below.\n"
    "Do NOT guess or add extra information.\n"
    "If answer is missing, say 'Not found in document'.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\nAnswer:"
)

# ⭐ Query
query_engine = index.as_query_engine(
    similarity_top_k=7,
    text_qa_template=qa_prompt
)

response = query_engine.query(
    "Quote exactly what Section 3 Performance Evaluation says in the document."
)

print("\n--- AI RESPONSE ---")
print(str(response))

# ⭐ Print citations (remove duplicates)
print("\n--- SOURCES ---")
seen = set()
for node in response.source_nodes:
    meta = tuple(node.metadata.items())
    if meta not in seen:
        print(node.metadata)
        seen.add(meta)
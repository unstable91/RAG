"""
One-time (re-runnable) ingestion: PDFs in ./data -> chunks -> embeddings -> Pinecone.

Usage:
    python ingest.py            # add/update documents (safe to re-run, no duplicates)
    python ingest.py --reset    # delete the index and rebuild from scratch
"""
import os
import sys
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
DATA_DIR = "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output size

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 60   # drops page headers / near-empty chunks
BATCH_SIZE = 100

if not PINECONE_API_KEY:
    sys.exit("PINECONE_API_KEY missing in .env")


def load_pdfs():
    docs = []
    pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        sys.exit(f"No PDFs found in ./{DATA_DIR}")
    for name in pdfs:
        print(f"Loading {name} ...")
        docs.extend(PyPDFLoader(os.path.join(DATA_DIR, name)).load())
    print(f"Loaded {len(docs)} pages.")
    return docs


def split(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs)
              if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    print(f"Created {len(chunks)} chunks.")
    return chunks


def ensure_index(pc, reset):
    existing = pc.list_indexes().names()
    if reset and INDEX_NAME in existing:
        print(f"Deleting index {INDEX_NAME} ...")
        pc.delete_index(INDEX_NAME)
        existing = []
    if INDEX_NAME not in existing:
        print(f"Creating index {INDEX_NAME} ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
    else:
        print(f"Using existing index {INDEX_NAME}.")


def main():
    reset = "--reset" in sys.argv
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, reset)

    chunks = split(load_pdfs())

    # Deterministic IDs -> re-running overwrites instead of duplicating
    counters = {}
    ids = []
    for c in chunks:
        key = (os.path.basename(str(c.metadata.get("source", ""))), c.metadata.get("page"))
        counters[key] = counters.get(key, 0) + 1
        ids.append(f"{key[0]}-p{key[1]}-c{counters[key]}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    for i in range(0, len(chunks), BATCH_SIZE):
        store.add_documents(chunks[i:i + BATCH_SIZE], ids=ids[i:i + BATCH_SIZE])
        print(f"Upserted {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print("Done.")


if __name__ == "__main__":
    main()

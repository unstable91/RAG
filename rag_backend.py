import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
TOP_K = 4

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Connecting to Pinecone index: {INDEX_NAME}")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

print(f"Loading Groq model: {GROQ_MODEL}")
llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are a medical question-answering assistant.

Answer the user's question using ONLY the retrieved context below.

Rules:
1. Use only the retrieved context. Do not add medical facts from your own knowledge.
2. If the answer is not in the context, say exactly:
   "I don't have enough information in the provided medical source to answer that."
3. Keep the answer concise and easy to understand.
4. Do not diagnose the user and do not prescribe medication or personal treatment plans.
5. For serious or emergency symptoms, advise consulting a medical professional.
6. Mention uncertainty when the source is incomplete.

Retrieved context:

{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{input}")]
)

# prompt -> llm -> string. Retrieval is done once, outside the chain.
answer_chain = prompt | llm | StrOutputParser()


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def extract_sources(docs):
    sources, seen = [], set()
    for doc in docs:
        meta = doc.metadata or {}
        source = os.path.basename(str(meta.get("source", "Medical source")))
        page = meta.get("page")
        try:
            display_page = int(page) + 1 if page is not None else meta.get("page_label")
        except (ValueError, TypeError):
            display_page = page
        key = (source, str(display_page))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": display_page})
    return sources


def ask_question(question):
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    question = question.strip()
    docs = retriever.invoke(question)  # single retrieval call

    answer = answer_chain.invoke(
        {"context": format_docs(docs), "input": question}
    )
    return {"answer": answer, "sources": extract_sources(docs)}

# Medical RAG (Flask + LangChain + Pinecone + Groq)

## Setup
1. python -m venv venv && source venv/bin/activate   (Windows: venv\Scripts\activate)
2. pip install -r requirements.txt
3. Copy .env.example to .env and fill in NEW API keys
4. Put Medical_book.pdf in ./data
5. python ingest.py          (use --reset to rebuild the index from scratch)
6. python app.py             -> http://127.0.0.1:5000

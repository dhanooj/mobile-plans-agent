import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    INDEX_NAME = "telco-knowledge-base"
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gemini-1.5-pro"

# Initialize Pinecone Client
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
pinecone_index = pc.Index(Config.INDEX_NAME)
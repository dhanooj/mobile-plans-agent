from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_core.tools import Tool
from config import Config, pinecone_index
import os
from dotenv import load_dotenv

load_dotenv()

def get_retrieval_tool():
    # Initialize Gemini for embeddings
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create embeddings using Gemini
    from langchain_core.embeddings import Embeddings
    
    class GeminiEmbeddings(Embeddings):
        def embed_documents(self, texts):
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                embeddings.append(result['embedding'])
            return embeddings
        
        def embed_query(self, text):
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
    
    embeddings = GeminiEmbeddings()
    
    vector_store = PineconeVectorStore(
        index=pinecone_index, 
        embedding=embeddings
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    def search_packages(query):
        """Search for telco packages matching the query"""
        docs = retriever.invoke(query)
        if not docs:
            return "No packages found matching your query."
        
        result = "Here are the relevant packages:\n\n"
        for doc in docs:
            result += f"- {doc.metadata.get('package_name', 'Package')}: {doc.metadata.get('description', doc.page_content[:100])}\n"
        return result
    
    return Tool(
        name="retrieve_plans",
        func=search_packages,
        description="Search for new package info, pricing, and data limits for Telco customers."
    )
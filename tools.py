from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_core.tools import Tool
from langchain_core.embeddings import Embeddings
from config import pinecone_index
import os

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Improved: Batching requests to Gemini API for better performance
        # Limits: Max 100 texts per call
        results = genai.embed_content(
            model="models/gemini-embedding-001", # Note: 'models/' prefix is often internal
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return results['embedding']
    
    def embed_query(self, text):
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="RETRIEVAL_QUERY"
        )
        return result['embedding']

def get_retrieval_tool():
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = GeminiEmbeddings()
    
    # Use the LangChain Pinecone integration
    vector_store = PineconeVectorStore(
        index=pinecone_index, 
        embedding=embeddings
    )
    
    # k=5 provides better context for the LLM to filter
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    def search_packages(query: str):
        """Search for telco packages matching the query"""
        docs = retriever.invoke(query)
        if not docs:
            return "No packages found matching your query."
        
        result = "Here are the relevant packages:\n\n"
        for doc in docs:
            result += f"- {doc.page_content[:100]}\n"
        return result

    return Tool(
        name="retrieve_plans",
        func=search_packages,
        description="Search for telco package details, pricing, and data limits. Useful for queries about reload amounts or data validity."
    )
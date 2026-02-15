# 📱 Mobile Package Listing Agent

An AI-powered agent that lists and answers questions about mobile data packages from a local service provider. The system uses scraped mobile plan data indexed in Pinecone and a LangGraph-orchestrated agent powered by Gemini LLM to generate intelligent responses. LangSmith is used for tracing and evaluation.

## 🚀 Features

- Scrapes mobile package data from a local service provider.
- Stores package embeddings in Pinecone vector database.
- Uses LangGraph to orchestrate an AI agent workflow.
- Gemini LLM generates contextual answers.
- Pinecone index used as retrieval tool.
- LangSmith tracing enabled for observability.
- LangSmith SDK used for automated evaluation.

## 🏗️ Project Structure
```python
.
├── packageScrape.ipynb   # Scrapes mobile packages and indexes them in Pinecone
├── graph.py              # LangGraph agent orchestration using Gemini LLM and Pinecone
├── tools.py              # Pinecone retrieval tool to list mobile packages
├── evaluator.ipynb       # LangSmith evaluation of agent responses
├── .env.example          # Sample environment variables file
├── requirements.txt      # Python dependencies
└── README.md
```


## ⚙️ Environment Variables

Configuration is managed using environment variables.

Create a .env file in the project root:
```python
PINECONE_API_KEY=""
PINECONE_ENV=""
PINECONE_INDEX=""
GOOGLE_API_KEY=""
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
```


## 🧠 Architecture
```python
User Query
    │
    ▼
LangGraph Agent (graph.py)
    │
    ├── Gemini LLM
    │
    └── Pinecone Retrieval Tool (tools.py)
            │
            ▼
        Pinecone Index(Scraped mobile plans from packageScrape.ipynb)


Tracing & Evaluation
    └── LangSmith
```

## 🧰 Tech Stack

- LangGraph
- Gemini LLM
- Pinecone Vector Database
- LangSmith
- Python
- Jupyter Notebook

## 📈 Future Improvements

- Improve the Pinecone index for better retrieval results
- Support multiple service providers (add more nodes to the LangGraph and conditional edges)
- Support multiple LLMs (add more nodes to the LangGraph and conditional edges)
- Add more criteria for evaluation


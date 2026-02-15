"""
Script to index Dialog postpaid packages into Pinecone using Gemini embeddings
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import hashlib

load_dotenv()

# Dialog postpaid packages data
DIALOG_PACKAGES = [
    {
        "name": "Power Plan 1300",
        "price": "Rs. 1300.00",
        "monthly": True,
        "features": [
            "Unlimited Any Net Voice",
            "1000 Dialog to Any Net SMS",
            "20GB Data with Data Rollover & Data Sharing",
            "Free ViU+ Subscription for 12 months"
        ],
        "description": "Affordable postpaid plan with unlimited calls and 20GB data"
    },
    {
        "name": "Power Plan 2100",
        "price": "Rs. 2100.00",
        "monthly": True,
        "features": [
            "Unlimited Any Net Voice",
            "1000 Dialog to Any Net SMS",
            "50GB Data with Data Rollover & Data Sharing",
            "Free ViU+ and Lionsgate Play Subscription for 12 months"
        ],
        "description": "Premium postpaid plan with unlimited calls and 50GB data plus entertainment subscriptions"
    },
    {
        "name": "Family Plans",
        "price": "Varies",
        "monthly": True,
        "features": [
            "Plans for up to 5 family members",
            "Shared data benefits",
            "Exclusive discounts on smartphones",
            "Special family rates"
        ],
        "description": "Exclusive family postpaid plans to keep your family connected with shared benefits"
    },
    {
        "name": "Smartphone Plan",
        "price": "40% upfront payment",
        "monthly": True,
        "features": [
            "Latest Android Smartphone",
            "Pay only 40% upfront",
            "Flexible postpaid plan",
            "Device financing included"
        ],
        "description": "Get the latest smartphone by paying only 40% upfront with postpaid plan"
    },
    {
        "name": "Friend Circle",
        "price": "Varies",
        "monthly": True,
        "features": [
            "Lifetime discounts for friends",
            "Special group rates",
            "Easy referral benefits",
            "Exclusive promotions"
        ],
        "description": "Enjoy lifetime discounts with your friend circle through Dialog's Friend Circle program"
    },
    {
        "name": "Dialog Prashansa Plans",
        "price": "Special rates",
        "monthly": True,
        "features": [
            "Exclusive plans for Government Pensioners",
            "Special pricing",
            "Premium benefits",
            "Dedicated support"
        ],
        "description": "Exclusive postpaid plans dedicated to Government Pensioners with special benefits"
    }
]

# Power Plan Benefits (common to all)
BENEFITS = [
    "Free Dialog Play & Lionsgate Play Subscription",
    "Data Rollover - Unused data carries to next month",
    "Data Share - Share data with loved ones for free",
    "Unlimited YouTube - Stream YouTube with just Rs. 120 add-on",
    "Loyalty Benefits - Exclusive discounts on smartphones",
    "Network Priority - High-priority network access"
]

def format_package_text(package):
    """Convert package dict to formatted text for embedding"""
    text = f"""
Package: {package['name']}
Price: {package['price']}
Description: {package['description']}

Features:
"""
    for feature in package['features']:
        text += f"- {feature}\n"
    
    text += "\nCommon Benefits:\n"
    for benefit in BENEFITS:
        text += f"- {benefit}\n"
    
    return text

def index_packages_to_pinecone():
    """Index Dialog packages to Pinecone using Gemini embeddings"""
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set")
        return
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, create if not
    index_name = "telco-knowledge-base"
    if index_name not in pc.list_indexes().names():
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=3072,  # Gemini embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    # Initialize Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("Error: GOOGLE_API_KEY not set")
        return
    
    genai.configure(api_key=gemini_key)
    
    # Process packages
    vectors = []
    for package in DIALOG_PACKAGES:
        text = format_package_text(package)
        
        # Generate embedding using Gemini
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = result['embedding']
        except Exception as e:
            print(f"Error embedding {package['name']}: {e}")
            continue
        
        # Create unique ID
        doc_id = f"dialog_{hashlib.md5(package['name'].encode()).hexdigest()[:8]}"
        
        # Prepare vector with metadata
        vector = {
            "id": doc_id,
            "values": embedding,
            "metadata": {
                "package_name": package['name'],
                "price": package['price'],
                "description": package['description'],
                "text": text[:1000]  # Store truncated text for reference
            }
        }
        vectors.append(vector)
        print(f"✓ Prepared: {package['name']}")
    
    # Upsert to Pinecone
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"\n✅ Successfully indexed {len(vectors)} packages to Pinecone!")
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
            return
    else:
        print("No vectors to index")

if __name__ == "__main__":
    print("Indexing Dialog postpaid packages to Pinecone...\n")
    index_packages_to_pinecone()

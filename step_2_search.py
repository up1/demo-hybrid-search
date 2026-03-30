import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

# --- CONFIGURATION ---
DB_PATH = "./product_search_db"
CSV_FILE = "products.csv"

def setup_db():
    # 1. Initialize OpenAI Embedding Function
    # 'text-embedding-3-small' is the 2026 standard for efficiency
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # 2. Initialize Chroma Client (Persistent)
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 3. Create or Get Collection
    collection = client.get_or_create_collection(
        name="products", 
        embedding_function=openai_ef
    )
    return collection

def hybrid_search(collection, query, category=None, n_results=3):
    """
    Simulates hybrid search by combining Semantic Similarity 
    with specific Metadata Filters.
    """
    where_filter = {}
    if category:
        where_filter["category"] = category

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter if where_filter else None
    )
    return results

if __name__ == "__main__":
    
    # Run Pipeline
    product_collection = setup_db()
    query = "latest apple smartphone"
    results = hybrid_search(product_collection, query, category="Mobile")
    print("\n--- Search Result ---")
    for i in range(len(results['ids'][0])):
        print(f"Match: {results['documents'][0][i]} | Metadata: {results['metadatas'][0][i]}")
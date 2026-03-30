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

    # 3. Remove existing collection for clean setup (Optional)
    if "products" in client.list_collections():
        client.delete_collection("products")
    
    # 4. Create or Get Collection
    collection = client.get_or_create_collection(
        name="products", 
        embedding_function=openai_ef
    )
    return collection

def ingest_data_from_csv(collection, csv_path):
    # Read CSV using Pandas
    df = pd.read_csv(csv_path)
    
    # Prepare data for Chroma
    # We use the product name for the 'document' (what gets embedded)
    documents = df["product_name"].tolist()
    ids = df["product_id"].astype(str).tolist()
    
    # Store all other columns as metadata for hybrid filtering
    metadatas = df.drop(columns=["product_name", "product_id"]).to_dict(orient="records")

    # Upsert (Insert or Update) into database
    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(documents)} products.")

if __name__ == "__main__":
    # Create a dummy CSV if it doesn't exist for testing
    if not os.path.exists(CSV_FILE):
        data = {
            "product_id": [101, 102, 103],
            "product_name": ["Apple iPhone 15 Pro", "Samsung S24 Ultra", "Sony Headphones"],
            "category": ["Mobile", "Mobile", "Audio"],
            "brand": ["Apple", "Samsung", "Sony"],
            "price": [999, 1199, 350]
        }
        pd.DataFrame(data).to_csv(CSV_FILE, index=False)

    # Run Pipeline
    product_collection = setup_db()
    ingest_data_from_csv(product_collection, CSV_FILE)
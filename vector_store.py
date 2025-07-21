import weaviate
from weaviate.connect import ConnectionParams
from weaviate.config import AdditionalConfig  # ✅ Modern usage
from sentence_transformers import SentenceTransformer
import torch
from config import WEAVIATE_URL
import numpy as np 

def store_embeddings(embeddings, transcriptions, file_ids, chunk_ids):
   

    # Initialize Weaviate v4 client with ConnectionParams.from_url
    connection_params = ConnectionParams.from_url(WEAVIATE_URL, grpc_port=50051)
    print("✅ Using vector_store.py at:", __file__)
    print("🛠 Using AdditionalConfig from:", AdditionalConfig)
    client = weaviate.WeaviateClient(
        connection_params=connection_params,
        additional_config=AdditionalConfig(),
        skip_init_checks=True  # Skip gRPC and other startup checks
    )
    client.connect() 
    # Create schema if not exists
    schema = {
        "class": "AudioEmbedding",
        "properties": [
            {"name": "transcription", "dataType": ["text"]},
            {"name": "file_id", "dataType": ["string"]},
            {"name": "chunk_id", "dataType": ["int"]},
        ],
        "vectorIndexConfig": {"pq": {"enabled": True}}
    }
    if not client.collections.exists("AudioEmbedding"):
        client.collections.create_from_dict(schema)
    
    # Batch insert embeddings
    collection = client.collections.get("AudioEmbedding")
    with collection.batch.dynamic() as batch:
        for emb, trans, fid, cid in zip(embeddings, transcriptions, file_ids, chunk_ids):
            if not trans.strip():
                print(f"[WARN] Skipping empty transcription at chunk {cid}")
                continue  # Skip blank/invalid texts

            if not isinstance(emb, (list, np.ndarray)):
                print(f"[WARN] Skipping invalid embedding at chunk {cid}")
                continue  # Skip invalid vectors
            try:
                vector = np.array(emb).astype(float).tolist()

                data_object = {
                    "transcription": trans,
                    "file_id": fid,
                    "chunk_id": cid
                }
                batch.add_object(properties=data_object, vector=vector)

            except Exception as e:
                print(f"[WARN] Skipping chunk {cid} due to bad vector or data: {e}")
                continue

        
    return client

def query_embeddings(query, client, n_results=5):
    text_model = SentenceTransformer("all-mpnet-base-v2", device="mps" if torch.backends.mps.is_available() else "cpu")
    query_embedding = text_model.encode([query])[0].tolist()
    
    collection = client.collections.get("AudioEmbedding")
    result = collection.query.near_vector(
        near_vector=query_embedding,
        limit=n_results,
        return_properties=["transcription", "file_id", "chunk_id"]
    )
    
    # Convert results to match the expected format
    metadatas = [{"transcription": obj.properties["transcription"], 
                  "file_id": obj.properties["file_id"], 
                  "chunk_id": obj.properties["chunk_id"]} for obj in result.objects]
    
    return {"metadatas": metadatas}
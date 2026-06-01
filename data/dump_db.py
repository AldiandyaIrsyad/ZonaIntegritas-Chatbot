import os
import pandas as pd
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def main():
    # Load env variables if .env exists
    # We navigate up one level from data/ to load the root .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)

    # Setup paths
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(DATA_DIR, exist_ok=True)
    
    md_output_path = os.path.join(DATA_DIR, "database_dump.md")
    md_content = ["# Database Dump\n"]

    # ----- POSTGRES DUMP -----
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    pg_url = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    engine = create_engine(pg_url)

    tables = [
        "pdf_documents",
        "parent_chunks",
        "ingestion_tasks",
        "sessions",
        "messages",
        "session_documents",
        "session_document_chunks"
    ]

    print("--- Dumping PostgreSQL Tables ---")
    md_content.append("## PostgreSQL Tables\n")
    
    for table in tables:
        try:
            df = pd.read_sql_table(table, engine)
            
            print(f"\n### Table: {table} ({len(df)} rows)")
            md_content.append(f"### Table: {table} ({len(df)} rows)\n")
            
            if df.empty:
                md_table = "*Table is empty*"
            else:
                md_table = df.to_markdown(index=False)
                
            print(md_table)
            md_content.append(md_table)
            md_content.append("\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to dump {table}: {e}")

    # ----- QDRANT DUMP -----
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

    print("\n--- Dumping Qdrant Collections ---")
    md_content.append("## Qdrant Collections\n")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections_response = client.get_collections()
        collections = [col.name for col in collections_response.collections]
        
        for collection in collections:
            try:
                points = []
                offset = None
                
                while True:
                    result, offset = client.scroll(
                        collection_name=collection,
                        limit=1000,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True
                    )
                    points.extend(result)
                    if offset is None:
                        break
                
                data = []
                for p in points:
                    row = {"id": p.id}
                    if p.payload:
                        row.update(p.payload)
                    # Exclude the actual vector by default to save space, but add dimensions if needed.
                    if p.vector:
                        if isinstance(p.vector, dict):
                            for k, v in p.vector.items():
                                row[f"vector_{k}"] = str(v)
                        else:
                            row["vector"] = str(p.vector)
                    data.append(row)
                    
                df = pd.DataFrame(data)
                
                print(f"\n### Qdrant Collection: {collection} ({len(df)} points)")
                md_content.append(f"### Qdrant Collection: {collection} ({len(df)} points)\n")
                
                if df.empty:
                    md_table = "*Collection is empty*"
                else:
                    md_table = df.to_markdown(index=False)
                    
                print(md_table)
                md_content.append(md_table)
                md_content.append("\n")
                
            except Exception as e:
                print(f"[ERROR] Failed to dump collection {collection}: {e}")
                
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")

    # Write to MD file
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"\n[OK] Saved all output to {md_output_path}")

if __name__ == "__main__":
    main()

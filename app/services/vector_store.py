from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.vector_chunk import VectorChunk
from typing import List, Tuple

async def store_chunks(db: Session, document_id: int, chunks: List[str], embeddings: List[List[float]]):
    """Store document chunks with embeddings in pgvector."""
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk = VectorChunk(
            document_id=document_id,
            chunk_text=chunk_text,
            chunk_index=idx,
            embedding=embedding,
        )
        db.add(chunk)
    db.commit()

async def search_similar_chunks(
    db: Session, 
    query_embedding: List[float], 
    limit: int = 5,
    similarity_threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """Search for similar chunks using cosine similarity."""
    # Convert embedding to PostgreSQL array format string
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Use raw SQL with proper parameter substitution
    query = f"""
        SELECT chunk_text, 1 - (embedding <=> '{embedding_str}'::vector) as similarity
        FROM vector_chunks
        WHERE 1 - (embedding <=> '{embedding_str}'::vector) >= {similarity_threshold}
        ORDER BY embedding <=> '{embedding_str}'::vector
        LIMIT {limit}
    """
    
    result = db.execute(text(query))
    
    chunks = []
    for row in result:
        chunks.append((row.chunk_text, float(row.similarity)))
    
    return chunks

async def delete_document_chunks(db: Session, document_id: int):
    """Delete all chunks for a document."""
    db.query(VectorChunk).filter(VectorChunk.document_id == document_id).delete()
    db.commit()


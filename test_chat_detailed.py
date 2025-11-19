"""
Detailed test to inspect chunk content and test with document-specific questions.
"""
import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.document import Document
from app.models.vector_chunk import VectorChunk
from app.services.embedding_service import get_embedding
from app.services.vector_store import search_similar_chunks

load_dotenv()

async def inspect_chunks():
    """Inspect the actual chunk content."""
    print("=" * 60)
    print("INSPECTING VECTOR CHUNKS")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        chunks = db.query(VectorChunk).all()
        print(f"\nTotal chunks: {len(chunks)}\n")
        
        for idx, chunk in enumerate(chunks, 1):
            print(f"Chunk {idx} (Document ID: {chunk.document_id}):")
            print("-" * 60)
            print(chunk.chunk_text)
            print()
        
        # Test with document-specific questions
        print("=" * 60)
        print("TESTING WITH DOCUMENT-SPECIFIC QUESTIONS")
        print("=" * 60)
        
        # Extract some keywords from chunks to form better questions
        keywords = []
        for chunk in chunks[:3]:  # Use first 3 chunks
            text = chunk.chunk_text.lower()
            if 'journey' in text:
                keywords.append('journey')
            if 'foreign key' in text or 'foreign' in text:
                keywords.append('foreign key')
            if 'table' in text:
                keywords.append('table')
            if 'phase' in text:
                keywords.append('phase')
        
        # Formulate better questions
        test_questions = [
            "What are the existing tables that need to be updated?",
            "Tell me about Phase 3 and foreign key constraints",
            "What new tables need to be created?",
            "What is the journeys table structure?",
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 60)
            
            try:
                question_embedding = await get_embedding(question)
                similar_chunks = await search_similar_chunks(
                    db, 
                    question_embedding, 
                    limit=3, 
                    similarity_threshold=0.65  # Lower threshold to get more results
                )
                
                if similar_chunks:
                    print(f"✅ Found {len(similar_chunks)} similar chunks:")
                    for idx, (chunk_text, similarity) in enumerate(similar_chunks, 1):
                        print(f"\n   Chunk {idx} [Similarity: {similarity:.3f}]:")
                        # Show first 200 chars of chunk
                        preview = chunk_text[:200].replace('\n', ' ')
                        print(f"   {preview}...")
                else:
                    print("⚠️  No similar chunks found")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(inspect_chunks())


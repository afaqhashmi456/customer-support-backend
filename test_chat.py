"""
Test script to verify chatbot functionality with uploaded PDFs.
This simulates the chat flow and tests RAG functionality.
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models.document import Document
from app.models.vector_chunk import VectorChunk
from app.services.embedding_service import get_embedding
from app.services.vector_store import search_similar_chunks
import httpx

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")

async def test_chatbot():
    """Test the chatbot RAG functionality."""
    print("=" * 60)
    print("CHATBOT TEST - RAG Functionality")
    print("=" * 60)
    
    # Check API key
    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY not set in .env file")
        return False
    
    print(f"✅ API Key: SET (length: {len(OPENROUTER_API_KEY)})")
    print()
    
    # Check database connection
    db = SessionLocal()
    try:
        # Check documents
        documents = db.query(Document).all()
        print(f"📄 Documents in database: {len(documents)}")
        
        if len(documents) == 0:
            print("⚠️  WARNING: No documents found in database!")
            print("   Please upload a PDF first using the /documents/upload endpoint")
            return False
        
        for doc in documents:
            chunk_count = db.query(VectorChunk).filter(VectorChunk.document_id == doc.id).count()
            print(f"   - {doc.filename} (ID: {doc.id}, Chunks: {chunk_count})")
        
        print()
        
        # Check vector chunks
        total_chunks = db.query(VectorChunk).count()
        print(f"📊 Total vector chunks: {total_chunks}")
        
        if total_chunks == 0:
            print("⚠️  WARNING: No vector chunks found!")
            print("   Documents may not have been processed correctly")
            return False
        
        print()
        
        # Test questions
        test_questions = [
            "What is this document about?",
            "Can you summarize the main points?",
            "Tell me about the key information in the document",
        ]
        
        print("🧪 Testing RAG queries...")
        print()
        
        for i, question in enumerate(test_questions, 1):
            print(f"Test {i}: {question}")
            print("-" * 60)
            
            try:
                # Step 1: Get embedding for question
                print("  1. Generating question embedding...")
                question_embedding = await get_embedding(question)
                print(f"     ✅ Embedding generated (dimension: {len(question_embedding)})")
                
                # Step 2: Search for similar chunks
                print("  2. Searching for similar chunks...")
                similar_chunks = await search_similar_chunks(
                    db, 
                    question_embedding, 
                    limit=5, 
                    similarity_threshold=0.7
                )
                
                if not similar_chunks:
                    print("     ⚠️  No similar chunks found (similarity threshold: 0.7)")
                    print("     💡 Try lowering the similarity threshold or asking a more specific question")
                    print()
                    continue
                
                print(f"     ✅ Found {len(similar_chunks)} similar chunks:")
                for idx, (chunk_text, similarity) in enumerate(similar_chunks, 1):
                    preview = chunk_text[:100].replace('\n', ' ') + "..." if len(chunk_text) > 100 else chunk_text
                    print(f"        {idx}. [Similarity: {similarity:.3f}] {preview}")
                
                # Step 3: Build context
                print("  3. Building context...")
                context = "\n\n".join([chunk[0] for chunk in similar_chunks])
                print(f"     ✅ Context built ({len(context)} characters)")
                
                # Step 4: Generate response using LLM
                print("  4. Generating response from LLM...")
                prompt = f"""Context from company documents:
{context}

Question: {question}

Answer based on the context above:"""
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "http://localhost:3000",
                            "X-Title": "AI Chatbot Test",
                        },
                        json={
                            "model": LLM_MODEL,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a helpful customer support assistant. Answer questions based on the provided context from company documents. If the context doesn't contain relevant information, politely say 'Sorry, I don't understand. Could you please rephrase your question or ask about something else?'"
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "stream": False,
                        },
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    ai_response = data["choices"][0]["message"]["content"]
                    print(f"     ✅ Response generated ({len(ai_response)} characters)")
                    print()
                    print("  🤖 AI Response:")
                    print("  " + "=" * 58)
                    # Indent the response
                    for line in ai_response.split('\n'):
                        print("  " + line)
                    print("  " + "=" * 58)
                    print()
                
            except ValueError as e:
                print(f"     ❌ Embedding error: {str(e)}")
                print()
                continue
            except ConnectionError as e:
                print(f"     ❌ Connection error: {str(e)}")
                print()
                continue
            except httpx.HTTPStatusError as e:
                print(f"     ❌ API error: {e.response.status_code}")
                print(f"     Response: {e.response.text}")
                print()
                continue
            except Exception as e:
                print(f"     ❌ Unexpected error: {str(e)}")
                import traceback
                traceback.print_exc()
                print()
                continue
        
        print("=" * 60)
        print("✅ TEST COMPLETED")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    success = asyncio.run(test_chatbot())
    sys.exit(0 if success else 1)


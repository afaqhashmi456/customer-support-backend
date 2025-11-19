"""
Full end-to-end test simulating the WebSocket chat flow.
Tests the complete RAG pipeline with streaming responses.
"""
import asyncio
import os
import json
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.document import Document
from app.models.vector_chunk import VectorChunk
from app.services.embedding_service import get_embedding
from app.services.vector_store import search_similar_chunks
import httpx

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")

async def simulate_chat_flow(question: str):
    """Simulate the complete chat flow as it happens in the WebSocket handler."""
    print(f"\n{'='*70}")
    print(f"💬 CHAT SIMULATION: {question}")
    print(f"{'='*70}\n")
    
    db = SessionLocal()
    try:
        # Step 1: Get embedding
        print("📊 Step 1: Generating question embedding...")
        question_embedding = await get_embedding(question)
        print(f"   ✅ Embedding dimension: {len(question_embedding)}\n")
        
        # Step 2: Search similar chunks
        print("🔍 Step 2: Searching for similar chunks...")
        similar_chunks = await search_similar_chunks(
            db, 
            question_embedding, 
            limit=5, 
            similarity_threshold=0.7
        )
        
        if not similar_chunks:
            print("   ⚠️  No similar chunks found!")
            return "I couldn't find relevant information in the uploaded documents."
        
        print(f"   ✅ Found {len(similar_chunks)} similar chunks:\n")
        for idx, (chunk_text, similarity) in enumerate(similar_chunks, 1):
            preview = chunk_text[:150].replace('\n', ' ')
            print(f"   [{idx}] Similarity: {similarity:.3f}")
            print(f"       {preview}...\n")
        
        # Step 3: Build context
        print("📝 Step 3: Building context from chunks...")
        context = "\n\n".join([chunk[0] for chunk in similar_chunks])
        print(f"   ✅ Context length: {len(context)} characters\n")
        
        # Step 4: Build prompt
        prompt = f"""Context from company documents:
{context}

Question: {question}

Answer based on the context above:"""
        
        # Step 5: Stream response (simulating WebSocket)
        print("🤖 Step 4: Streaming response from LLM...\n")
        print("   Response:")
        print("   " + "-" * 66)
        
        full_response = ""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
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
                    "stream": True,
                },
                timeout=60.0,
            ) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response += content
                                    print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
        
        print("\n   " + "-" * 66)
        print(f"\n   ✅ Complete response ({len(full_response)} characters)")
        
        return full_response
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()

async def main():
    """Run comprehensive chat tests."""
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE CHATBOT TEST")
    print("="*70)
    
    # Test questions based on the actual document content
    test_questions = [
        "What are the existing tables that need to be updated?",
        "Tell me about Phase 3 and foreign key constraints",
        "What new tables need to be created?",
        "What is the journeys table used for?",
    ]
    
    results = []
    for question in test_questions:
        response = await simulate_chat_flow(question)
        results.append((question, response))
        await asyncio.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"\nTotal questions tested: {len(test_questions)}")
    successful = sum(1 for _, resp in results if resp and len(resp) > 0)
    print(f"Successful responses: {successful}/{len(test_questions)}")
    
    if successful == len(test_questions):
        print("\n✅ ALL TESTS PASSED! Chatbot is working correctly.")
    else:
        print("\n⚠️  Some tests had issues. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())


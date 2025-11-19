import os
import httpx
from typing import List, Tuple
from dotenv import load_dotenv
from app.services.vector_store import search_similar_chunks
from app.services.embedding_service import get_embedding
from sqlalchemy.orm import Session

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

SYSTEM_PROMPT = """You are a helpful customer support assistant. Answer questions based on the provided context from company documents. If the context doesn't contain relevant information, politely say "Sorry, I don't understand. Could you please rephrase your question or ask about something else?""""

async def generate_rag_response(
    db: Session,
    question: str,
) -> str:
    """Generate RAG response using retrieved context."""
    # Get embedding for the question
    question_embedding = await get_embedding(question)
    
    # Search for similar chunks
    similar_chunks = await search_similar_chunks(
        db,
        question_embedding,
        limit=5,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    
    # If no similar chunks found, return default message
    if not similar_chunks:
        return "Sorry, I don't understand. Could you please rephrase your question or ask about something else?"
    
    # Build context from chunks
    context = "\n\n".join([chunk[0] for chunk in similar_chunks])
    
    # Build prompt
    prompt = f"""Context from company documents:
{context}

Question: {question}

Answer based on the context above:"""
    
    # Call OpenRouter LLM
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "AI Chatbot",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response


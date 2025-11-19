import os
import httpx
from typing import List
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-ada-002")

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts using OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")
    
    if not OPENROUTER_BASE_URL:
        raise ValueError("OPENROUTER_BASE_URL environment variable is not set")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional but recommended
        "X-Title": "ChatBot App",  # Optional but recommended
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/embeddings",
                headers=headers,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
    except httpx.ConnectError as e:
        raise ConnectionError(
            f"Failed to connect to OpenRouter API at {OPENROUTER_BASE_URL}. "
            f"Please check your internet connection and verify the API URL is correct. "
            f"Error: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"OpenRouter API returned an error: {e.response.status_code} - {e.response.text}"
        )

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text."""
    embeddings = await get_embeddings([text])
    return embeddings[0]


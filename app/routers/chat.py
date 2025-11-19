from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.routers.auth import get_current_user
from app.models.user import User
from app.models.chat import ChatHistory
from jose import jwt
from typing import List
from pydantic import BaseModel
import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")

router = APIRouter()

# Response models
class ChatHistoryItem(BaseModel):
    id: int
    message: str
    response: str
    created_at: str

    class Config:
        from_attributes = True

@router.get("/chat/history", response_model=List[ChatHistoryItem])
async def get_chat_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat history for the current user."""
    try:
        # Query chat history, ordered by newest first
        history = db.query(ChatHistory).filter(
            ChatHistory.user_id == current_user.id
        ).order_by(
            ChatHistory.created_at.desc()
        ).limit(limit).all()

        # Reverse to get oldest first (chronological order)
        history.reverse()

        # Convert created_at to ISO format string
        result = []
        for item in history:
            result.append({
                "id": item.id,
                "message": item.message,
                "response": item.response,
                "created_at": item.created_at.isoformat()
            })

        return result
    except Exception as e:
        print(f"[ChatHistory] Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

async def get_user_from_token(token: str, db: Session) -> User:
    """Verify token and get user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        user = db.query(User).filter(User.email == email).first()
        return user
    except (jwt.ExpiredSignatureError, jwt.JWTError, Exception):
        return None

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: str = Query(...)):
    """WebSocket endpoint for streaming chat with RAG."""
    await websocket.accept()

    from app.database import SessionLocal
    db = SessionLocal()

    try:
        user = await get_user_from_token(token, db)
        if not user:
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        while True:
            try:
                data = await websocket.receive_json()
                message = data.get("message", "")
            except WebSocketDisconnect:
                break
            except Exception as e:
                error_str = str(e)
                if "1001" in error_str or "1012" in error_str or "going away" in error_str.lower():
                    break
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Error receiving message: {str(e)}"
                    })
                except:
                    break
                continue
            
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "error": "Empty message"
                })
                continue

            if not OPENROUTER_API_KEY:
                await websocket.send_json({
                    "type": "error",
                    "error": "OpenRouter API key is not configured."
                })
                continue

            from app.services.embedding_service import get_embedding
            from app.services.vector_store import search_similar_chunks

            try:
                question_embedding = await get_embedding(message)
                similar_chunks = await search_similar_chunks(db, question_embedding, limit=5, similarity_threshold=0.7)
            except ValueError as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Embedding service error: {str(e)}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            except ConnectionError as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Connection error: {str(e)}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            except Exception as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Error getting embeddings: {str(e)}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            
            if not similar_chunks:
                context = ""
            else:
                context = "\n\n".join([chunk[0] for chunk in similar_chunks])

            if context:
                prompt = f"""Context from company documents:
{context}

Question: {message}

Answer based on the context above:"""
            else:
                prompt = message

            full_response = ""
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "http://localhost:3000",
                            "X-Title": "AI Chatbot",
                        },
                        json={
                            "model": LLM_MODEL,
                            "messages": [
                                {"role": "system", "content": """You are a friendly, warm, and helpful AI customer support assistant. You're here to have natural, human-like conversations with users.

GREETING RESPONSES:
- When users greet you (like 'hi', 'hello', 'hey', 'hye', 'sup', 'yo', 'greetings'), respond warmly and naturally like a human would. Examples:
  * "Hi there! How can I help you today?"
  * "Hello! I'm here to assist you. What can I do for you?"
  * "Hey! Great to hear from you. What's on your mind?"
  * "Hi! How are you doing? What can I help you with?"

THANK YOU RESPONSES:
- When users thank you (like 'thank you', 'thanks', 'thx', 'ty', 'appreciate it'), respond graciously and offer continued help. Examples:
  * "You're welcome! Happy to help. Is there anything else you need?"
  * "No problem at all! Let me know if you have any other questions."
  * "My pleasure! Feel free to ask if you need anything else."
  * "Anytime! I'm here if you need more help."
  * "Glad I could help! Don't hesitate to reach out again."

GOODBYE RESPONSES:
- When users say goodbye (like 'bye', 'goodbye', 'see ya', 'later', 'have a nice day'), respond warmly and wish them well. Examples:
  * "Goodbye! Have a great day!"
  * "See you later! Take care!"
  * "Bye! Feel free to come back anytime you need help."
  * "Have a wonderful day! I'm here whenever you need me."
  * "Take care! Don't hesitate to reach out if you need anything."

ANSWERING QUESTIONS:
- Answer questions based on the provided context from company documents when available
- If a question is about something not in the documents, be honest and helpful:
  * "I don't have that specific information in the documents I have access to. Is there something else I can help you with?"
  * "I'm not sure about that one. Could you ask me something else, or would you like me to help with a different topic?"

HANDLING UNCLEAR MESSAGES:
- If the question is unclear or you don't understand, say:
  * "I don't quite understand. Could you rephrase that for me?"
  * "Hmm, I'm not sure what you mean. Can you explain a bit more?"
  * "Sorry, I didn't catch that. Could you ask in a different way?"

TONE:
- Be conversational, friendly, and approachable
- Use casual language when appropriate
- Show empathy and understanding
- Don't be overly formal or robotic
- Use natural transitions and connecting words"""},
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
                                            await websocket.send_json({
                                                "type": "message",
                                                "content": content,
                                            })
                                except json.JSONDecodeError:
                                    continue
            except httpx.ConnectError as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to connect to API. Error: {str(e)}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            except httpx.HTTPStatusError as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"API error: {e.response.status_code}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            except Exception as e:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unexpected error: {str(e)}"
                    })
                    await websocket.send_json({"type": "done"})
                except:
                    pass
                continue
            
            try:
                await websocket.send_json({"type": "done"})
            except:
                break

            try:
                chat_history = ChatHistory(
                    user_id=user.id,
                    message=message,
                    response=full_response,
                )
                db.add(chat_history)
                db.commit()
            except:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
            })
        except:
            pass
    finally:
        try:
            db.close()
        except:
            pass


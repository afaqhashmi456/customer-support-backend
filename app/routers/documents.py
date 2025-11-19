from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.models.document import Document
from app.models.vector_chunk import VectorChunk
from app.routers.auth import get_current_admin_user, get_current_user
from app.models.user import User
from app.utils.pdf_parser import extract_text_from_pdf
from app.utils.text_splitter import split_text
from app.services.embedding_service import get_embeddings
from app.services.vector_store import store_chunks, delete_document_chunks
import os
import uuid
from pydantic import BaseModel

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class DocumentResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: str
    chunk_count: int

    class Config:
        from_attributes = True

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Upload and process a document (admin only)."""
    # Validate file type
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are supported"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Extract text
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file_content)
    else:
        text = file_content.decode('utf-8')
    
    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File appears to be empty or could not extract text"
        )
    
    # Save file
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    # Split text into chunks
    chunks = split_text(text)
    
    # Generate embeddings
    embeddings = await get_embeddings(chunks)
    
    # Create document record
    document = Document(
        filename=file.filename,
        file_path=file_path,
        uploaded_by=current_user.id,
        chunk_count=len(chunks),
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Store chunks with embeddings
    await store_chunks(db, document.id, chunks, embeddings)
    
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        uploaded_at=document.uploaded_at.isoformat(),
        chunk_count=document.chunk_count,
    )

@router.get("/list", response_model=List[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """List all uploaded documents (admin only)."""
    documents = db.query(Document).all()
    return [
        DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            uploaded_at=doc.uploaded_at.isoformat(),
            chunk_count=doc.chunk_count,
        )
        for doc in documents
    ]

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Delete a document (admin only)."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete file
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete chunks
    await delete_document_chunks(db, document_id)
    
    # Delete document record
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}


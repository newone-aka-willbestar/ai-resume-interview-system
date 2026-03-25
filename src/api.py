from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import logging

from src.rag_chain import RAGChain
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("api")

app = FastAPI(title="华科制造智能售后客服系统", version="2.1.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="无效的 API Key")
    return True

vector_manager = VectorStoreManager()

def get_rag_chain():
    """每次请求刷新 RAG 链（解决全局单例 Bug）"""
    return RAGChain(vector_manager=vector_manager)

class QuestionRequest(BaseModel):
    question: str
    use_hyde: Optional[bool] = True

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.get("/")
async def root():
    return {"message": "华科制造智能售后客服系统 API 已启动", "version": "2.1.0"}

@app.post("/ask", response_model=AnswerResponse, dependencies=[Depends(verify_api_key)])
async def ask(request: QuestionRequest, rag_chain: RAGChain = Depends(get_rag_chain)):
    try:
        result = rag_chain.answer(request.question)
        return AnswerResponse(**result)
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail="服务内部错误")

@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    if file.size > 10 * 1024 * 1024:  # 10MB 限制
        raise HTTPException(status_code=400, detail="文件过大（最大10MB）")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        processor = DocumentProcessor()
        docs = processor.process(tmp_path)

        vector_manager.add_documents(docs)   # 关键：使用单例增量添加

        os.unlink(tmp_path)
        logger.info(f"文档 {file.filename} 上传成功，处理 {len(docs)} 个块")
        return {"message": f"上传成功！已添加 {len(docs)} 个文本块", "filename": file.filename}
    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.OLLAMA_MODEL, "vector_db": vector_manager.get_vectorstore() is not None}
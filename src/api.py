from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

from src.rag_chain import RAGChain
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.config import Config

app = FastAPI(title="AI智能客服系统", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
rag_chain = RAGChain()

class QuestionRequest(BaseModel):
    question: str
    use_hyde: Optional[bool] = True

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.get("/")
async def root():
    return {"message": "AI智能客服系统API", "status": "running"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """问答接口"""
    try:
        # 临时设置HyDE开关
        if not request.use_hyde:
            original = Config.USE_HYDE
            Config.USE_HYDE = False
            result = rag_chain.answer(request.question)
            Config.USE_HYDE = original
        else:
            result = rag_chain.answer(request.question)
        
        return AnswerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档接口"""
    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 处理文档
        processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        docs = processor.process(tmp_path)
        
        # 更新向量库
        vector_manager = VectorStoreManager()
        vector_manager.create_from_documents(docs)
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        return {
            "message": f"文档上传成功，已处理 {len(docs)} 个文本块",
            "chunks": len(docs),
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
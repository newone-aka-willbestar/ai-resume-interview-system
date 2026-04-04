import os
import tempfile
import logging
import time
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag import RAG
from src.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="华科制造 AI 智能客服")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAG() # 全局 RAG 实例

class QuestionRequest(BaseModel):
    question: str

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="无效 API Key")

@app.post("/upload")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="只支持 PDF")
    
    # 步骤 1：安全写入临时文件
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await file.read())
        
        # 步骤 2：加载并处理
        loader = DocumentLoader()
        docs = loader.load_and_split(tmp_path)
        
        vector_store = VectorStore()
        vector_store.add_documents(docs)
        
        # 更新 RAG 检索器
        rag.init_retriever(docs)
        
        return {"message": f"成功处理 {len(docs)} 个文本块", "filename": file.filename}
    
    except Exception as e:
        logger.error(f"❌ 上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 步骤 3：强制清理，解决 Windows 占用问题
        gc.collect()
        time.sleep(0.2) # 给系统毫秒级的释放时间
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

@app.post("/ask")
async def ask(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    try:
        return rag.ask(request.question)
    except Exception as e:
        return {"answer": f"系统繁忙，请稍后再试: {str(e)}", "sources": []}
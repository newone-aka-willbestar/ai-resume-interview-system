from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from src.config import settings

class DocumentProcessor:
    """处理文档加载与分割（使用 config 参数）"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ('.txt', '.md'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
    
    def process(self, file_path: str) -> List[Document]:
        docs = self.load_documents(file_path)
        return self.split_documents(docs)
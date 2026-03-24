from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Optional
from src.config import Config

class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(self, persist_directory: str = Config.VECTORSTORE_PATH):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.vectorstore = None
    
    def create_from_documents(self, documents: List[Document]):
        """从文档创建向量库"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        return self.vectorstore
    
    def load_existing(self):
        """加载已有向量库"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vectorstore
    
    def get_retriever(self, top_k: int = Config.TOP_K):
        """获取检索器"""
        if self.vectorstore is None:
            self.load_existing()
        
        # 使用MMR（最大边际相关性）检索，避免结果重复
        return self.vectorstore.as_retriever(
            search_type="mmr",  # 也可用 "similarity"
            search_kwargs={"k": top_k, "fetch_k": top_k * 2}
        )
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
from src.config import settings
import logging
import os

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or settings.VECTORSTORE_PATH
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.vectorstore = None

    def create_from_documents(self, documents: List[Document]):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info(f"向量库创建成功，添加 {len(documents)} 个文档块")
        return self.vectorstore

    def load_existing(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
        return self.vectorstore

    def get_retriever(self, top_k: int = None):
        if self.vectorstore is None:
            self.load_existing()
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k or settings.TOP_K, "fetch_k": (top_k or settings.TOP_K) * 2}
        )
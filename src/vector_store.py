from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from typing import List
from src.config import Config
import os

class VectorStoreManager:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or Config.VECTORSTORE_PATH
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vectorstore = None

    def create_from_documents(self, documents: List[Document]):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectorstore

    def load_existing(self):
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            # 首次运行时创建空库
            self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_directory)
        return self.vectorstore

    def get_retriever(self, top_k: int = Config.TOP_K):
        if self.vectorstore is None:
            self.load_existing()
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 2}
        )
import os
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 环境变量：国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # 1. 使用绝对路径计算
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
        
        # 2. 检查模型关键文件是否存在
        is_ready = os.path.exists(local_model_path) and os.path.isfile(os.path.join(local_model_path, "vocab.txt"))

        if is_ready:
            model_name = local_model_path
            logger.info(f"✅ 成功定位本地 Embedding 模型: {model_name}")
        else:
            model_name = "BAAI/bge-small-zh-v1.5"
            logger.warning(f"⚠️ 本地模型缺失，正在通过镜像站自动下载...")

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"❌ 模型加载终极失败: {e}")
            raise RuntimeError("模型初始化失败，请检查网络或 models 目录。")

        self.persist_directory = os.path.join(base_dir, "chroma_db")
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> Optional[Chroma]:
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return None

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents, 
                embedding=self.embeddings, 
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        logger.info(f"✅ 已持久化存储 {len(documents)} 个文本块")

    def get_retriever(self):
        if self.vectorstore is None:
            self.vectorstore = self._load_vectorstore()
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}) if self.vectorstore else None
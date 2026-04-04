import os
import shutil
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import settings

# 设置国内镜像站，防止联网下载模型时卡死（面试加分点：考虑到国内开发环境）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # 1. 动态计算绝对路径，防止 Windows 相对路径偏移
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(base_dir, "models", "bge-small-zh-v1.5")
        
        # 2. 核心校验：不但检查文件夹，还要检查关键文件 vocab.txt 是否存在
        # 报错 TypeError 就是因为文件夹在，但 vocab.txt 缺失
        vocab_file = os.path.join(local_model_path, "vocab.txt")
        
        if os.path.exists(local_model_path) and os.path.isfile(vocab_file):
            model_name = local_model_path
            logger.info(f"✅ 找到完整的本地模型，正在加载: {model_name}")
        else:
            # 如果本地不完整，直接使用模型名，HuggingFace 库会自动下载到缓存
            model_name = "BAAI/bge-small-zh-v1.5"
            logger.warning(f"⚠️ 本地模型不完整或不存在，将尝试自动在线加载/补全: {model_name}")

        try:
            # 3. 加载模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                # 强制使用 CPU，避免小白没有显卡驱动导致初始化失败
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"❌ Embedding 加载失败: {e}")
            # 如果联网也失败，给出清晰的提示
            raise RuntimeError(f"无法初始化嵌入模型。请检查网络或手动下载模型到 {local_model_path}")
        
        # 向量库存储路径
        self.persist_directory = os.path.join(base_dir, "chroma_db")
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> Optional[Chroma]:
        """尝试从本地加载已有的向量库"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"📦 加载本地向量数据库: {self.persist_directory}")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None

    def add_documents(self, documents: List[Document]):
        """增量添加文档"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        logger.info(f"✅ 成功添加 {len(documents)} 个文本块到向量库")

    def get_retriever(self):
        """采用 MMR 策略的检索器"""
        if self.vectorstore is None:
            self.vectorstore = self._load_vectorstore()
            if self.vectorstore is None:
                logger.warning("⚠️ 向量库为空，请先上传文档")
                return None
        
        # 从配置中读取 TOP_K，体现工程化思维
        k = getattr(settings, "TOP_K", 4)
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            }
        )

    def clear_db(self):
        """清空向量库（方便演示）"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.vectorstore = None
            logger.info("🧹 已清理向量库缓存")